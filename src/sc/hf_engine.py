"""HF generation with an arbitrary KV prefix (full spec cache, Attention-Matching compacted cache,
or a trained cartridge). Built on the Attention-Matching repo's modified Qwen3 model and
CompactedPrefixCache so that conditions 2/3/4 share one code path.

Prefix file format (torch.save dict):
  layers: list over layers of (K (1,KV,t,D) bf16, beta (1,KV,t) bf16, V (1,KV,t,D) bf16)
  original_seq_len: int  -> new tokens get RoPE positions starting here (rope_base = orig - t)
  yarn_factor: float|None -> the model must be loaded with the same rope scaling
  meta: dict
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
COMPACTION_DIR = ROOT / "third_party" / "compaction"
if str(COMPACTION_DIR) not in sys.path:
    sys.path.insert(0, str(COMPACTION_DIR))

from .rollouts import MODEL_NAME, SAMPLING  # noqa: E402

NATIVE_MAX_POS = 32768


def is_qwen(name: str) -> bool:
    return "qwen" in name.lower()


def load_model(yarn_factor: float | None = None, device: str = "cuda", model_name: str | None = None):
    """Qwen3: same rope scaling as the vLLM condition-2 run (yarn x4, original 32768).
    Mistral (Nemo): loaded through the compaction repo's Llama model class with a LlamaConfig built from
    the Mistral config (identical weight names; head_dim/rope_theta carried over), no rope scaling."""
    from transformers import AutoTokenizer
    model_name = model_name or MODEL_NAME
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"device_map": device, "dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if is_qwen(model_name):
        from models.qwen3 import Qwen3ForCausalLM
        if yarn_factor:
            kwargs["rope_scaling"] = {"rope_type": "yarn", "factor": float(yarn_factor),
                                      "original_max_position_embeddings": NATIVE_MAX_POS}
            kwargs["max_position_embeddings"] = int(NATIVE_MAX_POS * yarn_factor)
        model = Qwen3ForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        from transformers import AutoConfig, LlamaConfig
        from models.llama import LlamaForCausalLM
        assert not yarn_factor, "rope scaling only implemented for Qwen3"
        d = AutoConfig.from_pretrained(model_name).to_dict()
        for k in ("model_type", "architectures", "sliding_window"):
            d.pop(k, None)
        model = LlamaForCausalLM.from_pretrained(model_name, config=LlamaConfig(**d), **kwargs)
    model.eval()
    print(f"loaded {model_name} rope_scaling={kwargs.get('rope_scaling')}", flush=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def _render(tok, system_prompt, user_prompt):
    return tok.apply_chat_template([{"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=SAMPLING["enable_thinking"])


def render_system_prefix(tok, system_prompt: str) -> str:
    """The cacheable prefix shared by every request with this system prompt.
    Qwen3: the rendered system turn '<|im_start|>system\\n...<|im_end|>\\n' (unchanged from the pilot caches).
    Other templates (Mistral puts the system text inside the first [INST] block): the longest common
    string prefix of two renderings with different user prompts, backed off to a clean token boundary."""
    if is_qwen(tok.name_or_path):
        return tok.apply_chat_template([{"role": "system", "content": system_prompt}], tokenize=False,
                                       add_generation_prompt=False, enable_thinking=SAMPLING["enable_thinking"])
    a, b = _render(tok, system_prompt, "hello one"), _render(tok, system_prompt, "zzz two")
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    while i > 0:
        pre, suf = a[:i], a[i:]
        if tok(a, add_special_tokens=False).input_ids == (tok(pre, add_special_tokens=False).input_ids
                                                           + tok(suf, add_special_tokens=False).input_ids):
            return pre
        i -= 1
    raise ValueError("could not find a clean prefix/suffix token boundary")


def render_user_suffix(tok, system_prompt: str, user_prompt: str) -> str:
    """Everything after the system prefix: user turn + assistant generation prompt."""
    full = _render(tok, system_prompt, user_prompt)
    pre = render_system_prefix(tok, system_prompt)
    assert full.startswith(pre), "chat template prefix mismatch"
    return full[len(pre):]


def spec_token_indices(tok, system_prefix: str, spec_text: str) -> range:
    """Token index range of the raw spec text inside the rendered system prefix (for compaction:
    template + preamble tokens are kept verbatim, only spec tokens are compacted)."""
    start_char = system_prefix.index(spec_text)
    a = len(tok(system_prefix[:start_char], add_special_tokens=False).input_ids)
    b = len(tok(system_prefix[:start_char + len(spec_text)], add_special_tokens=False).input_ids)
    return range(a, b)


@torch.inference_mode()
def prefill_prefix(model, tok, prefix_text: str, chunk_size: int = 4096):
    """Run the model over the prefix and return (DynamicCache, n_tokens)."""
    from models.generate import chunked_prefill
    from transformers.cache_utils import DynamicCache
    ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    cache = DynamicCache()
    chunked_prefill(model, input_ids=ids, past_key_values=cache, chunk_size=chunk_size)
    return cache, ids.shape[1]


def cache_to_prefix_tuple(cache) -> tuple:
    """DynamicCache -> ((K, beta=0, V), ...) as expected by CompactedPrefixCache."""
    out = []
    for layer in cache.layers:
        k, v = layer.keys, layer.values
        beta = torch.zeros(k.shape[0], k.shape[1], k.shape[2], dtype=k.dtype, device=k.device)
        out.append((k, beta, v))
    return tuple(out)


def prefix_num_bytes(layers) -> int:
    return sum(t.numel() * t.element_size() for L in layers for t in L)


def save_prefix(path, layers, original_seq_len: int, yarn_factor, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"layers": [tuple(t.detach().to("cpu", torch.bfloat16).contiguous() for t in L) for L in layers],
                "original_seq_len": int(original_seq_len), "yarn_factor": yarn_factor, "meta": meta}, path)


def load_prefix(path, device="cuda"):
    d = torch.load(path, map_location=device)
    d["layers"] = tuple(tuple(t.to(device) for t in L) for L in d["layers"])
    return d


@torch.inference_mode()
def generate_with_prefix(model, tok, prefix_layers, original_seq_len: int, suffixes: list[str],
                         batch_size: int = 16, max_new_tokens: int | None = None, verbose: bool = True):
    """Generate for a list of already-rendered suffix strings. Returns list of dicts."""
    from models.generate import generate_with_compacted_cache_batch
    max_new_tokens = max_new_tokens or SAMPLING["max_new_tokens"]
    outs = []
    # Sort by length to reduce padding, then restore order.
    order = sorted(range(len(suffixes)), key=lambda i: len(suffixes[i]))
    t_all = time.time()
    for bi in range(0, len(order), batch_size):
        idx = order[bi:bi + batch_size]
        torch.manual_seed(SAMPLING["seed"] + bi)
        t0 = time.time()
        texts = generate_with_compacted_cache_batch(
            model=model, tokenizer=tok, prompts=[suffixes[i] for i in idx], compacted_cache=prefix_layers,
            max_new_tokens=max_new_tokens, temperature=SAMPLING["temperature"], top_k=SAMPLING["top_k"],
            top_p=SAMPLING["top_p"], original_seq_len=original_seq_len)
        dt = time.time() - t0
        for i, t in zip(idx, texts):
            n_out = len(tok(t, add_special_tokens=False).input_ids)
            outs.append((i, {"text": t, "n_output_tokens": n_out,
                             "finish_reason": "length" if n_out >= max_new_tokens - 1 else "stop",
                             "batch_wall_s": dt, "batch_size": len(idx)}))
        if verbose:
            print(f"  batch {bi // batch_size + 1}/{(len(order) + batch_size - 1) // batch_size}: {dt:.1f}s "
                  f"({time.time() - t_all:.0f}s total)", flush=True)
    outs.sort(key=lambda x: x[0])
    return [o for _, o in outs]


@torch.inference_mode()
def measure_latency(model, tok, prefix_layers, original_seq_len: int, suffixes: list[str],
                    max_new_tokens: int = 128) -> dict:
    """Single-request latency (batch=1) and peak memory for a handful of prompts."""
    from models.generate import generate_with_compacted_cache_batch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    times = []
    # warmup
    generate_with_compacted_cache_batch(model=model, tokenizer=tok, prompts=suffixes[:1], compacted_cache=prefix_layers,
                                        max_new_tokens=8, temperature=SAMPLING["temperature"], top_k=SAMPLING["top_k"],
                                        top_p=SAMPLING["top_p"], original_seq_len=original_seq_len)
    for s in suffixes:
        torch.cuda.synchronize()
        t0 = time.time()
        generate_with_compacted_cache_batch(model=model, tokenizer=tok, prompts=[s], compacted_cache=prefix_layers,
                                            max_new_tokens=max_new_tokens, temperature=SAMPLING["temperature"],
                                            top_k=SAMPLING["top_k"], top_p=SAMPLING["top_p"],
                                            original_seq_len=original_seq_len)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return {"n": len(times), "max_new_tokens": max_new_tokens, "latency_s_mean": sum(times) / len(times),
            "latency_s_min": min(times), "latency_s_max": max(times),
            "ms_per_token_mean": 1000 * sum(times) / len(times) / max_new_tokens,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "base_alloc_gb": base_alloc / 1e9,
            "prefix_kv_gb": prefix_num_bytes(prefix_layers) / 1e9, "prefix_tokens": prefix_layers[0][0].shape[2]}
