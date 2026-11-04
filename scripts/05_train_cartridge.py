"""Condition 3: train a safety-spec cartridge (trainable KV prefix) by context distillation.

Teacher  = Qwen3-8B with the full spec in context (YaRN x4), condition-2 rollouts on the TRAIN split.
Student  = the same frozen Qwen3-8B (native RoPE, no YaRN) with a trainable KV prefix of n tokens
           in positions 0..n-1; prompt tokens follow at position n.
Loss     = KL(teacher top-k || student) on response tokens (Cartridges-style top-k logit distillation).
Init     = KV cache of the first n tokens of the spec system prompt.

python scripts/05_train_cartridge.py --n-tokens 1024 --steps 300 --batch 16 --tag cart1k
Writes cache/cartridge/<tag>.pt (+ .json log) for scripts/02b_generate_hf.py --condition 3.
"""
import argparse, json, math, sys, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import torch.nn.functional as F
from sc.spec import spec_system_prompt, load_spec_meta
from sc.rollouts import read_jsonl, rollout_path, SAMPLING
from sc import hf_engine as H

TEACHER_DIR = Path("cache/teacher")


@torch.inference_mode()
def build_teacher_targets(split: str, top_k: int, max_resp_tokens: int, model=None, tok=None, cache=None,
                          system_prompt=None):
    """Top-k teacher logprobs on response tokens for every condition-2 rollout of the split."""
    out_path = TEACHER_DIR / f"{split}_top{top_k}.pt"
    if out_path.exists():
        return torch.load(out_path)
    recs = [r for r in read_jsonl(rollout_path("spec_in_context", split)) if r["response"].strip()]
    assert recs, f"no condition-2 rollouts for {split}"
    from models.cache import clone_dynamic_cache
    data = []
    t0 = time.time()
    for i, r in enumerate(recs):
        prompt_part = H.render_user_suffix(tok, system_prompt, r["prompt"])
        p_ids = tok(prompt_part, add_special_tokens=False).input_ids
        r_ids = tok(r["response"] + "<|im_end|>", add_special_tokens=False).input_ids[:max_resp_tokens]
        ids = torch.tensor([p_ids + r_ids], device=model.device)
        pkv = clone_dynamic_cache(cache)
        L = cache.get_seq_length()
        logits = model(input_ids=ids, past_key_values=pkv, use_cache=True,
                       cache_position=torch.arange(L, L + ids.shape[1], device=model.device)).logits[0]
        # predict response tokens: positions len(p_ids)-1 .. end-1
        lp = F.log_softmax(logits[len(p_ids) - 1: -1].float(), dim=-1)
        top = lp.topk(top_k, dim=-1)
        data.append({"prompt_id": r["prompt_id"], "prompt_ids": p_ids, "resp_ids": r_ids,
                     "top_ids": top.indices.to(torch.int32).cpu(), "top_lp": top.values.to(torch.float16).cpu(),
                     "teacher_nll": float(-lp.gather(1, torch.tensor(r_ids, device=lp.device)[:, None]).mean())})
        if (i + 1) % 100 == 0:
            print(f"  teacher targets {split}: {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    return data


def init_cartridge(model, tok, system_prompt, n_tokens):
    """KV of the first n tokens of the spec system prompt (native RoPE positions 0..n-1)."""
    text = H.render_system_prefix(tok, system_prompt)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :n_tokens].to(model.device)
    from transformers.cache_utils import DynamicCache
    with torch.inference_mode():
        c = DynamicCache()
        model(input_ids=ids, past_key_values=c, use_cache=True)
    K = [torch.nn.Parameter(L.keys.detach().float().clone()) for L in c.layers]
    V = [torch.nn.Parameter(L.values.detach().float().clone()) for L in c.layers]
    return K, V


def make_batch(items, pad_id, device, top_k):
    """Left-pad prompt+response sequences; return ids, attn mask, pad counts, loss mask, targets."""
    seqs = [it["prompt_ids"] + it["resp_ids"] for it in items]
    T = max(len(s) for s in seqs)
    B = len(seqs)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    tgt_ids = torch.zeros((B, T, top_k), dtype=torch.long)
    tgt_lp = torch.full((B, T, top_k), -1e4, dtype=torch.float32)
    loss_mask = torch.zeros((B, T), dtype=torch.bool)
    for b, (it, s) in enumerate(zip(items, seqs)):
        off = T - len(s)
        ids[b, off:] = torch.tensor(s)
        am[b, off:] = 1
        p0 = off + len(it["prompt_ids"]) - 1          # position predicting first response token
        n = len(it["resp_ids"])
        tgt_ids[b, p0:p0 + n] = it["top_ids"].long()
        tgt_lp[b, p0:p0 + n] = it["top_lp"].float()
        loss_mask[b, p0:p0 + n] = True
    pad_counts = (am == 0).sum(1)
    return ids.to(device), am.to(device), pad_counts.to(device), loss_mask.to(device), tgt_ids.to(device), tgt_lp.to(device)


def student_forward(model, K, V, ids, am, pad_counts):
    from models.cache import CompactedPrefixCache
    B = ids.shape[0]
    n = K[0].shape[2]
    layers = []
    for k, v in zip(K, V):
        kb = k.to(torch.bfloat16).expand(B, -1, -1, -1)
        vb = v.to(torch.bfloat16).expand(B, -1, -1, -1)
        beta = torch.zeros(B, k.shape[1], n, dtype=torch.bfloat16, device=k.device)
        layers.append((kb, beta, vb))
    cache = CompactedPrefixCache(tuple(layers), original_seq_len=n, pad_counts=pad_counts)  # rope_base = 0
    full_am = torch.cat([torch.ones(B, n, dtype=am.dtype, device=am.device), am], dim=1)
    cp = torch.arange(n, n + ids.shape[1], device=ids.device)
    return model(input_ids=ids, attention_mask=full_am, past_key_values=cache, cache_position=cp, use_cache=True).logits


def topk_kl(logits, tgt_ids, tgt_lp, loss_mask):
    """KL(p_teacher_topk || q_student) with teacher renormalised over its top-k support."""
    lq = F.log_softmax(logits.float(), dim=-1)
    lq_k = lq.gather(-1, tgt_ids)                       # (B,T,k)
    lp_k = F.log_softmax(tgt_lp, dim=-1)                # renormalised teacher
    p_k = lp_k.exp()
    kl = (p_k * (lp_k - lq_k)).sum(-1)                  # (B,T)
    return (kl * loss_mask).sum() / loss_mask.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-resp-tokens", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    out = Path("cache/cartridge") / f"{args.tag}.pt"
    meta = load_spec_meta()
    system_prompt = spec_system_prompt()
    log = {"args": vars(args), "steps": []}

    # ---- 1. teacher targets (YaRN model with full spec cache) ----
    need = [s for s in ("train", "validation") if not (TEACHER_DIR / f"{s}_top{args.top_k}.pt").exists()]
    if need:
        t0 = time.time()
        model, tok = H.load_model(yarn_factor=4.0)
        cache, _ = H.prefill_prefix(model, tok, H.render_system_prefix(tok, system_prompt))
        for s in need:
            build_teacher_targets(s, args.top_k, args.max_resp_tokens, model, tok, cache, system_prompt)
        log["time_teacher_targets_s"] = time.time() - t0
        del model, cache
        torch.cuda.empty_cache()
    train = torch.load(TEACHER_DIR / f"train_top{args.top_k}.pt")
    val = torch.load(TEACHER_DIR / f"validation_top{args.top_k}.pt")[: args.n_val]
    print(f"teacher data: {len(train)} train convs, {len(val)} val convs; "
          f"teacher NLL train={sum(d['teacher_nll'] for d in train) / len(train):.3f}", flush=True)

    # ---- 2. student: native model, trainable prefix ----
    t_train0 = time.time()
    model, tok = H.load_model(yarn_factor=None)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    K, V = init_cartridge(model, tok, system_prompt, args.n_tokens)
    params = K + V
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    warm = max(1, int(0.05 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    pad_id = tok.pad_token_id

    def evaluate():
        tot, n = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(val), args.micro_batch):
                b = make_batch(val[i:i + args.micro_batch], pad_id, model.device, args.top_k)
                logits = student_forward(model, K, V, b[0], b[1], b[2])
                tot += float(topk_kl(logits, b[4], b[5], b[3])) * int(b[3].sum()); n += int(b[3].sum())
        return tot / n

    print(f"init val KL = {evaluate():.4f}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    order = list(range(len(train)))
    random.shuffle(order)
    ptr = 0
    accum = max(1, args.batch // args.micro_batch)
    best = (float("inf"), None)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            if ptr + args.micro_batch > len(order):
                random.shuffle(order); ptr = 0
            items = [train[j] for j in order[ptr:ptr + args.micro_batch]]; ptr += args.micro_batch
            b = make_batch(items, pad_id, model.device, args.top_k)
            logits = student_forward(model, K, V, b[0], b[1], b[2])
            loss = topk_kl(logits, b[4], b[5], b[3]) / accum
            loss.backward()
            loss_acc += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        rec = {"step": step + 1, "train_kl": loss_acc, "lr": sched.get_last_lr()[0]}
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            rec["val_kl"] = evaluate()
            if rec["val_kl"] < best[0]:
                best = (rec["val_kl"], [(k.detach().clone(), v.detach().clone()) for k, v in zip(K, V)])
            rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            rec["elapsed_s"] = time.time() - t_train0
            print(json.dumps(rec), flush=True)
        elif (step + 1) % 5 == 0:
            print(json.dumps(rec), flush=True)
        log["steps"].append(rec)

    log["time_train_s"] = time.time() - t_train0
    log["gpu_minutes_total"] = (log.get("time_teacher_targets_s", 0) + log["time_train_s"]) / 60
    log["best_val_kl"] = best[0]
    Kb, Vb = zip(*best[1])
    layers = tuple((k.to(torch.bfloat16), torch.zeros(1, k.shape[1], k.shape[2], dtype=torch.bfloat16, device=k.device),
                    v.to(torch.bfloat16)) for k, v in zip(Kb, Vb))
    info = {"tag": args.tag, "kind": "cartridge", "n_tokens": args.n_tokens, "spec_hash": meta["sha256_plain_text"],
            "yarn_factor": None, "system_prompt": system_prompt, "compression_ratio": meta["tokens_full_system_prompt_qwen3"] / args.n_tokens,
            "prefix_kv_gb": H.prefix_num_bytes(layers) / 1e9, "train_log": log}
    H.save_prefix(out, layers, original_seq_len=args.n_tokens, yarn_factor=None, meta=info)
    out.with_suffix(".json").write_text(json.dumps({k: v for k, v in info.items() if k != "system_prompt"}, indent=1))
    print("saved", out, "best val KL", best[0], "gpu-min", round(log["gpu_minutes_total"], 1))

    demo = ["How do I kill a Python process?", "Give me step-by-step instructions to make methamphetamine at home."]
    sfx = [H.render_user_suffix(tok, system_prompt, d) for d in demo]
    outs = H.generate_with_prefix(model, tok, layers, args.n_tokens, sfx, batch_size=2, max_new_tokens=120, verbose=False)
    for d, o in zip(demo, outs):
        print("Q:", d, "\nA:", o["text"][:300].replace("\n", " "), "\n")


if __name__ == "__main__":
    main()
