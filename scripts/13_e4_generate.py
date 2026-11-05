"""E4 generation for one condition on CoSApien test prompts (or generated train/val prompts, or the jailbreak set).

In context (vLLM):   --system none | policy:game_development | stack:game_development | policy:a+b | stack:a+b | spec
Cartridges (HF):     --prefixes cache/cartridge/cart1k.pt,cache/cartridge/pol_game.pt   (KV concatenated in that order)
LoRA (vLLM):         --lora cache/lora/<adapter>   (short "helpful assistant" system prompt)

python scripts/13_e4_generate.py --name game_incontext --system policy:game_development --rows test
Output: data/rollouts/e4/<name>.jsonl (resumable).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from sc import e4
from sc.rollouts import read_jsonl, write_jsonl, SAMPLING
from sc.spec import NO_SPEC_SYSTEM_PROMPT


def rows_for(which, pols):
    if which == "jb":
        from sc.data import load_prompts
        return [{"id": r["id"], "policy": None, "type": "harmful_jb", "prompt": r["prompt"]}
                for r in load_prompts("validation", prompt_set="jailbreak")]
    return e4.load_rows(which, pols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rows", default="test", choices=["test", "train", "val", "jb"])
    ap.add_argument("--policies", default=",".join(e4.MAIN_POLICIES), help="which policies' prompts to run")
    ap.add_argument("--system", default=None)
    ap.add_argument("--prefixes", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.thinking:
        SAMPLING.update({"enable_thinking": True, "max_new_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20})
    else:
        SAMPLING["max_new_tokens"] = 768
    out = e4.rollout_file(args.name)
    rows = rows_for(args.rows, args.policies.split(","))[: args.limit]
    done = {r["prompt_id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.name}: {len(rows)} rows, {len(todo)} to generate", flush=True)
    if not todo:
        return
    prompts = [r["prompt"] for r in todo]
    t0 = time.time()
    if args.prefixes:
        from sc import hf_engine as H
        parts = [H.load_prefix(p) for p in args.prefixes.split(",")]
        assert all(not p["yarn_factor"] for p in parts)
        layers = tuple(tuple(torch.cat([p["layers"][l][j] for p in parts], dim=2) for j in range(3))
                       for l in range(len(parts[0]["layers"])))
        orig = sum(p["layers"][0][0].shape[2] for p in parts)
        model, tok = H.load_model(yarn_factor=None)
        sfx = [H.render_user_suffix(tok, NO_SPEC_SYSTEM_PROMPT, p) for p in prompts]
        bs = 64 if not args.thinking else 48
        outs = H.generate_with_prefix(model, tok, layers, orig, sfx, batch_size=bs)
        texts = [o["text"] for o in outs]
        meta = {"engine": "hf-prefix", "prefixes": args.prefixes, "prefix_tokens": orig}
        fins = [o["finish_reason"] for o in outs]
    else:
        from sc.vllm_engine import build_llm, chat_generate
        kind = args.system or "none"
        sp = e4.system_prompt_for(kind)
        long = kind == "spec" or kind.startswith("stack:")
        llm = build_llm(yarn_factor=4.0 if long else None, gpu_memory_utilization=0.85,
                        max_model_len=None if long else 16384, enable_lora=bool(args.lora))
        outs, _ = chat_generate(llm, sp, prompts, lora_path=args.lora)
        texts = [o["text"] for o in outs]
        fins = [o["finish_reason"] for o in outs]
        meta = {"engine": "vllm", "system": kind, "lora": args.lora}
    wall = time.time() - t0
    recs = read_jsonl(out)
    for r, t, f in zip(todo, texts, fins):
        resp = t.split("</think>")[-1].strip() if args.thinking else t
        recs.append({"prompt_id": r["id"], "policy": r.get("policy"), "type": r.get("type"), "tag": r.get("tag"),
                     "prompt": r["prompt"], "response": resp, "raw_response": t if args.thinking else None,
                     "condition": args.name, "finish_reason": f, "thinking": args.thinking,
                     "sampling": dict(SAMPLING), **meta})
    write_jsonl(out, recs)
    print(f"done {len(todo)} in {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
