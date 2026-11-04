"""Query-vector generation for Attention Matching, tailored to the safety-spec use case.

Two sources, both only using TRAIN-split data:
  * 'self'  : queries of the spec tokens themselves (the spec studies itself; no generation needed)
  * 'train' : queries of (user prompt + condition-2 response) tokens from TRAIN rollouts, computed with the
              full spec in the KV cache -> exactly the attention pattern seen at inference time.
Output format matches the compaction repo's QueryGenerator: (num_layers, num_kv_heads, n_per_kv_head, head_dim).
"""
import torch


def _extractor(model, tok):
    from compaction.query_generation.self_study import SelfStudyQueryGenerator
    return SelfStudyQueryGenerator(model=model, tokenizer=tok, config=None, verbose=False)


def _attention_to_kv(q, num_kv_heads):
    from compaction.query_generation.query_generator import QueryGenerator
    L, A, n, D = q.shape
    return QueryGenerator._attention_to_kv(q, num_kv_heads=num_kv_heads, heads_per_kv_head=A // num_kv_heads)


@torch.inference_mode()
def self_queries(model, tok, prefix_ids: torch.Tensor, indices: range, max_tokens: int = 20000, seed: int = 0):
    """Queries from the spec tokens during a fresh prefill of the prefix (positions in `indices`)."""
    from transformers.cache_utils import DynamicCache
    ex = _extractor(model, tok)
    head_dim = model.config.head_dim
    q = ex._extract_query_vectors_from_prefill(input_ids=prefix_ids, past_key_values=DynamicCache(),
                                               head_dim=head_dim, start_token_idx=None)  # (L, A, T, D)
    q = q[:, :, indices.start:indices.stop, :]
    if q.shape[2] > max_tokens:
        g = torch.Generator(device="cpu").manual_seed(seed)
        sel = torch.randperm(q.shape[2], generator=g)[:max_tokens].sort().values.to(q.device)
        q = q[:, :, sel, :]
    return q.contiguous()


@torch.inference_mode()
def train_queries(model, tok, full_cache, suffix_texts: list[str], max_tokens_per_conv: int = 768):
    """Queries from prompt+response tokens of train conversations, attending to the full spec cache."""
    ex = _extractor(model, tok)
    head_dim = model.config.head_dim
    qs = []
    for i, s in enumerate(suffix_texts):
        ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_tokens_per_conv].to(model.device)
        q = ex._extract_query_vectors_from_prefill(input_ids=ids, past_key_values=full_cache, head_dim=head_dim,
                                                   start_token_idx=0)
        qs.append(q)
        if (i + 1) % 16 == 0:
            print(f"    train queries: {i + 1}/{len(suffix_texts)} convs, {sum(x.shape[2] for x in qs)} tokens", flush=True)
    return torch.cat(qs, dim=2).contiguous()


def to_kv_space(q_attention, num_kv_heads, max_per_kv_head: int | None = None, seed: int = 0):
    q = _attention_to_kv(q_attention, num_kv_heads)
    if max_per_kv_head and q.shape[2] > max_per_kv_head:
        g = torch.Generator(device="cpu").manual_seed(seed)
        sel = torch.randperm(q.shape[2], generator=g)[:max_per_kv_head].to(q.device)
        q = q[:, :, sel, :]
    return q.contiguous()


class PrecomputedQueryGenerator:
    """Drop-in replacement for compaction.query_generation.QueryGenerator returning fixed queries."""
    queries = None
    stats = None

    def __init__(self, *args, **kwargs):
        pass

    def generate_queries(self, *args, **kwargs):
        return PrecomputedQueryGenerator.queries, PrecomputedQueryGenerator.stats, None


def install_precomputed(queries, stats):
    import compaction.query_generation as qg
    PrecomputedQueryGenerator.queries = queries
    PrecomputedQueryGenerator.stats = {"final_n_queries_per_kv_head": queries.shape[2], "methods_used": {}, **stats}
    qg.QueryGenerator = PrecomputedQueryGenerator
