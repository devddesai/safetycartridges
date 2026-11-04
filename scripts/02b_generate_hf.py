"""Generate rollouts with the HF prefix-cache engine.

  --condition 2 : rebuild the full-spec prefix on the fly (YaRN x4); used as a cross-engine check
  --condition 3 : --prefix cache/cartridge/<run>.pt
  --condition 4 : --prefix cache/compaction/<run>.pt
  --condition 1 : no prefix beyond the short system prompt (cross-engine check)

Usage: python scripts/02b_generate_hf.py --condition 4 --prefix cache/compaction/x.pt --split validation [--limit 50] [--tag x]
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from sc.data import load_prompts
from sc.spec import spec_system_prompt, load_spec_meta, NO_SPEC_SYSTEM_PROMPT
from sc.rollouts import CONDITIONS, rollout_path, make_record, read_jsonl, write_jsonl
from sc import hf_engine as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prompt-set", default="main", choices=["main", "jailbreak"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--latency-n", type=int, default=8, help="prompts for batch-1 latency measurement")
    args = ap.parse_args()

    name = CONDITIONS[args.condition]
    if args.prompt_set != "main":
        args.tag = (args.tag + "_" if args.tag else "") + "jb"
    tag = args.tag or (Path(args.prefix).stem if args.prefix else "hf")
    out_path = rollout_path(name, args.split, tag)
    rows = load_prompts(args.split, limit=args.limit, prompt_set=args.prompt_set)
    done = {r["prompt_id"] for r in read_jsonl(out_path)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{name}/{args.split}/{tag}: {len(rows)} prompts, {len(done)} done, {len(todo)} to generate", flush=True)

    meta = load_spec_meta()
    spec_hash = meta["sha256_plain_text"] if args.condition != 1 else None
    if args.condition in (3, 4):
        pre = H.load_prefix(args.prefix)
        yarn = pre["yarn_factor"]
        model, tok = H.load_model(yarn_factor=yarn)
        layers, orig_len = pre["layers"], pre["original_seq_len"]
        system_prompt = pre["meta"].get("system_prompt") or spec_system_prompt()
        prefix_meta = {k: v for k, v in pre["meta"].items() if k not in ("system_prompt", "train_log")}
    elif args.condition == 2:
        yarn = 4.0
        model, tok = H.load_model(yarn_factor=yarn)
        system_prompt = spec_system_prompt()
        cache, orig_len = H.prefill_prefix(model, tok, H.render_system_prefix(tok, system_prompt))
        layers = H.cache_to_prefix_tuple(cache)
        prefix_meta = {"kind": "full_spec_kv"}
    else:
        yarn = None
        model, tok = H.load_model(yarn_factor=None)
        system_prompt = NO_SPEC_SYSTEM_PROMPT
        cache, orig_len = H.prefill_prefix(model, tok, H.render_system_prefix(tok, system_prompt))
        layers = H.cache_to_prefix_tuple(cache)
        prefix_meta = {"kind": "short_system_prompt_kv"}
    t = layers[0][0].shape[2]
    bs = args.batch_size or max(1, min(32, int(60e9 // max(1, H.prefix_num_bytes(layers)))))
    print(f"prefix tokens={t} ({H.prefix_num_bytes(layers) / 1e9:.2f} GB), original_seq_len={orig_len}, "
          f"yarn={yarn}, batch_size={bs}", flush=True)

    suffixes = [H.render_user_suffix(tok, system_prompt, r["prompt"]) for r in todo]
    cost = {"condition": name, "tag": tag, "split": args.split, "engine": "hf_prefix", "prefix_tokens": t,
            "prefix_kv_gb": H.prefix_num_bytes(layers) / 1e9, "yarn_factor": yarn, "prefix_meta": prefix_meta}
    if args.latency_n and suffixes:
        cost["latency"] = H.measure_latency(model, tok, layers, orig_len, suffixes[:args.latency_n])
        print(json.dumps(cost["latency"], indent=1), flush=True)
    if suffixes:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        outs = H.generate_with_prefix(model, tok, layers, orig_len, suffixes, batch_size=bs)
        wall = time.time() - t0
        recs = read_jsonl(out_path)
        for r, o in zip(todo, outs):
            recs.append(make_record(r, args.split, name, o["text"], engine="hf_prefix(compaction-repo qwen3, sdpa)",
                                    spec_hash=spec_hash,
                                    extra={"finish_reason": o["finish_reason"], "n_output_tokens": o["n_output_tokens"],
                                           "yarn_factor": yarn, "prefix_tag": tag, "prefix_tokens": t}))
        write_jsonl(out_path, recs)
        cost.update({"n_prompts": len(todo), "wall_s": wall, "s_per_prompt_batched": wall / len(todo),
                     "batch_size": bs, "peak_mem_gb_batched": torch.cuda.max_memory_allocated() / 1e9,
                     "output_tokens": sum(o["n_output_tokens"] for o in outs)})
    Path("results").mkdir(exist_ok=True)
    Path(f"results/cost_{name}_{tag}_{args.split}.json").write_text(json.dumps(cost, indent=1))
    print(json.dumps({k: v for k, v in cost.items() if k != "prefix_meta"}, indent=1))


if __name__ == "__main__":
    main()
