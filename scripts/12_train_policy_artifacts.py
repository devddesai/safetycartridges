"""E4: distil one deployment policy into (a) a cartridge and (b) a LoRA, from the SAME teacher targets.

Teacher = Qwen3-8B with the policy in context (rollouts from 13_e4_generate.py --rows train/val --system policy:<P>,
named teacher_<short>_train / teacher_<short>_val). Targets = top-20 teacher logprobs on response tokens.
Cartridge: n trainable KV tokens initialised from the first n tokens of the policy system prompt.
LoRA: delegated to 11_train_lora.py with the same target files.

python scripts/12_train_policy_artifacts.py --policy game_development [--n-tokens 256 --steps 200]
"""
import argparse, importlib.util, json, math, random, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import torch.nn.functional as F
from sc import e4
from sc import hf_engine as H
from sc.rollouts import read_jsonl

_spec = importlib.util.spec_from_file_location("tc", Path(__file__).with_name("05_train_cartridge.py"))
tc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tc)
TEACHER_DIR = Path("cache/teacher")


@torch.inference_mode()
def build_targets(model, tok, system_prompt, recs, top_k=20, max_resp=512):
    from transformers.cache_utils import DynamicCache
    from models.cache import clone_dynamic_cache
    cache, L = H.prefill_prefix(model, tok, H.render_system_prefix(tok, system_prompt))
    data = []
    for r in recs:
        p_ids = tok(H.render_user_suffix(tok, system_prompt, r["prompt"]), add_special_tokens=False).input_ids
        r_ids = tok(r["response"] + "<|im_end|>", add_special_tokens=False).input_ids[:max_resp]
        ids = torch.tensor([p_ids + r_ids], device=model.device)
        logits = model(input_ids=ids, past_key_values=clone_dynamic_cache(cache), use_cache=True,
                       cache_position=torch.arange(L, L + ids.shape[1], device=model.device)).logits[0]
        lp = F.log_softmax(logits[len(p_ids) - 1:-1].float(), dim=-1)
        top = lp.topk(top_k, dim=-1)
        data.append({"prompt_id": r["prompt_id"], "prompt_ids": p_ids, "resp_ids": r_ids,
                     "top_ids": top.indices.to(torch.int32).cpu(), "top_lp": top.values.to(torch.float16).cpu(),
                     "teacher_nll": float(-lp.gather(1, torch.tensor(r_ids, device=lp.device)[:, None]).mean())})
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--skip-lora", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0); random.seed(0)
    short = e4.SHORT[args.policy]
    sp = e4.policy_system_prompt([args.policy])
    tfiles = {s: TEACHER_DIR / f"pol_{short}_{s}_top20.pt" for s in ("train", "val")}
    model, tok = H.load_model(yarn_factor=None)
    t0 = time.time()
    for s, f in tfiles.items():
        if not f.exists():
            recs = [r for r in read_jsonl(e4.rollout_file(f"teacher_{short}_{s}")) if r["response"].strip()]
            assert recs, f"missing teacher rollouts teacher_{short}_{s}"
            TEACHER_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(build_targets(model, tok, sp, recs), f)
            print(f"targets {s}: {len(recs)} convs", flush=True)
    t_targets = time.time() - t0
    train, val = torch.load(tfiles["train"]), torch.load(tfiles["val"])[:64]

    out = Path("cache/cartridge") / f"pol_{short}.pt"
    if not out.exists():
        for p in model.parameters():
            p.requires_grad_(False)
        n_sys = len(tok(H.render_system_prefix(tok, sp), add_special_tokens=False).input_ids)
        n = min(args.n_tokens, n_sys)
        K, V = tc.init_cartridge(model, tok, sp, n)
        opt = torch.optim.AdamW(K + V, lr=args.lr, weight_decay=0.0)
        warm = max(1, int(0.05 * args.steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
        pad = tok.pad_token_id

        def evaluate():
            tot, m = 0.0, 0
            with torch.inference_mode():
                for i in range(0, len(val), args.micro_batch):
                    b = tc.make_batch(val[i:i + args.micro_batch], pad, model.device, 20)
                    tot += float(tc.topk_kl(tc.student_forward(model, K, V, *b[:3]), b[4], b[5], b[3])) * int(b[3].sum()); m += int(b[3].sum())
            return tot / m
        log = {"policy": args.policy, "n_tokens": n, "policy_system_tokens": n_sys, "init_val_kl": evaluate(), "steps": []}
        print(f"cartridge n={n} (policy prompt {n_sys} tokens), init val KL {log['init_val_kl']:.4f}", flush=True)
        order, ptr, accum = list(range(len(train))), 0, args.batch // args.micro_batch
        random.shuffle(order)
        best, t1 = (float("inf"), None), time.time()
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                if ptr + args.micro_batch > len(order):
                    random.shuffle(order); ptr = 0
                b = tc.make_batch([train[j] for j in order[ptr:ptr + args.micro_batch]], pad, model.device, 20); ptr += args.micro_batch
                (tc.topk_kl(tc.student_forward(model, K, V, *b[:3]), b[4], b[5], b[3]) / accum).backward()
            torch.nn.utils.clip_grad_norm_(K + V, 1.0); opt.step(); sched.step()
            if (step + 1) % 25 == 0 or step + 1 == args.steps:
                v = evaluate(); log["steps"].append({"step": step + 1, "val_kl": v})
                print(json.dumps(log["steps"][-1]), flush=True)
                if v < best[0]:
                    best = (v, [(k.detach().clone(), vv.detach().clone()) for k, vv in zip(K, V)])
        layers = tuple((k.to(torch.bfloat16), torch.zeros(1, k.shape[1], k.shape[2], dtype=torch.bfloat16, device=k.device),
                        v.to(torch.bfloat16)) for k, v in best[1])
        log.update({"best_val_kl": best[0], "train_s": time.time() - t1, "targets_s": t_targets})
        H.save_prefix(out, layers, original_seq_len=n, yarn_factor=None, meta={"kind": "policy_cartridge", **log})
        out.with_suffix(".json").write_text(json.dumps(log, indent=1))
        print("saved", out, "best val KL", round(best[0], 4), flush=True)
    del model; torch.cuda.empty_cache()

    if not args.skip_lora and not Path(f"cache/lora/lora_{short}").exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("11_train_lora.py")), "--tag", f"lora_{short}",
                        "--train", str(tfiles["train"]), "--val", str(tfiles["val"]), "--steps", str(args.steps)], check=True)


if __name__ == "__main__":
    main()
