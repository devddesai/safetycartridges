"""E3 / E4: LoRA context distillation with EXACTLY the cartridge's data and objective.

Teacher targets = the same top-k teacher logprob files the cartridge trains on (cache/teacher/*.pt, built by
05_train_cartridge.py or 12_policy_targets.py). Student = base model + LoRA, with the short no-spec system prompt
("You are a helpful assistant.") instead of the spec / cartridge. Loss = KL(teacher top-k || student) on response
tokens, same batch size / steps / schedule shape as the cartridge (only the learning rate differs: LoRA-standard).

python scripts/11_train_lora.py --tag lora_spec_r16 [--train cache/teacher/train_top20.pt --val cache/teacher/validation_top20.pt]
Writes cache/lora/<tag>/ (peft adapter, servable by vLLM --enable-lora) + <tag>.json log.
"""
import argparse, importlib.util, json, math, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from sc.spec import NO_SPEC_SYSTEM_PROMPT
from sc.rollouts import MODEL_NAME

_spec = importlib.util.spec_from_file_location("tc", Path(__file__).with_name("05_train_cartridge.py"))
tc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tc)  # make_batch, topk_kl
from sc import hf_engine as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train", default="cache/teacher/train_top20.pt")
    ap.add_argument("--val", default="cache/teacher/validation_top20.pt")
    ap.add_argument("--system-prompt", default=NO_SPEC_SYSTEM_PROMPT)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA trainable params: {n_params / 1e6:.1f}M ({n_params * 2 / 1e9:.3f} GB bf16)", flush=True)

    pre = tok(H.render_system_prefix(tok, args.system_prompt), add_special_tokens=False).input_ids
    def with_pre(items):
        return [{**it, "prompt_ids": pre + list(it["prompt_ids"])} for it in items]
    train = with_pre(torch.load(args.train))
    val = with_pre(torch.load(args.val)[: args.n_val])
    print(f"{len(train)} train / {len(val)} val conversations", flush=True)
    pad_id = tok.pad_token_id or tok.eos_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = max(1, int(0.05 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    def fwd(b):
        ids, am = b[0], b[1]
        pos = (am.cumsum(1) - 1).clamp(min=0)
        return model(input_ids=ids, attention_mask=am, position_ids=pos).logits

    def evaluate():
        model.eval(); tot, n = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(val), 8):
                b = tc.make_batch(val[i:i + 8], pad_id, model.device, args.top_k)
                tot += float(tc.topk_kl(fwd(b), b[4], b[5], b[3])) * int(b[3].sum()); n += int(b[3].sum())
        model.train(); return tot / n

    log = {"args": vars(args), "trainable_params": n_params, "steps": []}
    log["init_val_kl"] = evaluate(); print("init val KL", round(log["init_val_kl"], 4), flush=True)
    torch.cuda.reset_peak_memory_stats()
    order = list(range(len(train))); random.shuffle(order); ptr = 0
    accum = max(1, args.batch // args.micro_batch)
    best, out = float("inf"), Path("cache/lora") / args.tag
    t_train0 = time.time()
    model.train()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True); la = 0.0
        for _ in range(accum):
            if ptr + args.micro_batch > len(order):
                random.shuffle(order); ptr = 0
            b = tc.make_batch([train[j] for j in order[ptr:ptr + args.micro_batch]], pad_id, model.device, args.top_k)
            ptr += args.micro_batch
            loss = tc.topk_kl(fwd(b), b[4], b[5], b[3]) / accum
            loss.backward(); la += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        rec = {"step": step + 1, "train_kl": la}
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            rec["val_kl"] = evaluate(); rec["elapsed_s"] = time.time() - t_train0
            rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            if rec["val_kl"] < best:
                best = rec["val_kl"]; model.save_pretrained(out)
            print(json.dumps(rec), flush=True)
        elif (step + 1) % 10 == 0:
            print(json.dumps(rec), flush=True)
        log["steps"].append(rec)
    log.update({"best_val_kl": best, "time_train_s": time.time() - t_train0, "time_total_s": time.time() - t0,
                "gpu_minutes_train": (time.time() - t_train0) / 60, "adapter_gb_bf16": n_params * 2 / 1e9})
    Path(f"cache/lora/{args.tag}.json").write_text(json.dumps(log, indent=1))
    print("saved", out, "best val KL", best, "train min", round(log["gpu_minutes_train"], 1), flush=True)


if __name__ == "__main__":
    main()
