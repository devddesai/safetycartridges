"""Generate rollouts for conditions 1 (no spec) and 2 (full spec in context) with vLLM.

Usage: python scripts/02_generate.py --condition 2 --split validation [--limit 50]
Writes data/rollouts/<name>_<split>.jsonl (resumable: already-generated prompt ids are skipped)
and results/cost_<name>_<split>.json.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from sc.data import load_prompts
from sc.spec import spec_system_prompt, load_spec_meta, NO_SPEC_SYSTEM_PROMPT
from sc.rollouts import CONDITIONS, rollout_path, make_record, read_jsonl, write_jsonl
from sc.vllm_engine import build_llm, chat_generate



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=int, required=True, choices=[1, 2])
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--preamble", default=None, help="spec preamble variant (see sc.spec.PREAMBLES)")
    args = ap.parse_args()

    name = CONDITIONS[args.condition]
    out_path = rollout_path(name, args.split, args.tag)
    rows = load_prompts(args.split, limit=args.limit)
    done = {r["prompt_id"] for r in read_jsonl(out_path)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{name}/{args.split}: {len(rows)} prompts, {len(done)} already done, {len(todo)} to generate")
    if not todo:
        return

    if args.condition == 2:
        meta = load_spec_meta()
        system_prompt, spec_hash, yarn = spec_system_prompt(preamble=args.preamble), meta["sha256_plain_text"], 4.0
    else:
        system_prompt, spec_hash, yarn = NO_SPEC_SYSTEM_PROMPT, None, None

    llm = build_llm(yarn_factor=yarn, gpu_memory_utilization=args.gpu_mem)
    torch.cuda.reset_peak_memory_stats()
    outs, wall = chat_generate(llm, system_prompt, [r["prompt"] for r in todo])
    recs = read_jsonl(out_path)
    for r, o in zip(todo, outs):
        recs.append(make_record(r, args.split, name, o["text"], engine="vllm-0.11.0", spec_hash=spec_hash,
                                extra={"finish_reason": o["finish_reason"], "n_output_tokens": o["n_output_tokens"],
                                       "n_prompt_tokens": o["n_prompt_tokens"],
                                       "yarn_factor": yarn, "preamble": args.preamble if args.condition == 2 else None}))
    write_jsonl(out_path, recs)
    n_out = sum(o["n_output_tokens"] for o in outs)
    cost = {"condition": name, "split": args.split, "engine": "vllm", "n_prompts": len(todo),
            "wall_s": wall, "s_per_prompt_batched": wall / len(todo), "output_tokens": n_out,
            "output_tok_per_s": n_out / wall, "prompt_tokens_mean": sum(o["n_prompt_tokens"] for o in outs) / len(outs),
            "finish_reasons": {k: sum(o["finish_reason"] == k for o in outs) for k in {o["finish_reason"] for o in outs}}}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/cost_{name}{('_' + args.tag) if args.tag else ''}_{args.split}.json").write_text(json.dumps(cost, indent=1))
    print(json.dumps(cost, indent=1))


if __name__ == '__main__':
    main()
