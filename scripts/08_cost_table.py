"""Cost table: prefix KV memory, batch-1 latency (HF, same engine for every condition), one-time cost."""
import json, glob
from pathlib import Path

R = Path("results")
rows = []

def lat(path):
    p = R / path
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return d.get("latency", {}) | {"peak_mem_gb_batched": d.get("peak_mem_gb_batched"), "s_per_prompt_batched": d.get("s_per_prompt_batched")}

# C1 / C2 batch-1 latency measured with the HF engine for comparability (see run below), C3/C4 from their evals.
entries = [
    ("C1 no spec", "cost_no_spec_hf_validation.json", 0, 0.0, "-"),
    ("C2 full spec (YaRN x4)", "cost_spec_in_context_hf_validation.json", 55749, 8.22, "prefill 11 s per request-set (cached)"),
    ("C4 AM 4096", "cost_attention_matching_am4096_validation.json", None, None, "cache/compaction/am4096.json"),
    ("C4 AM 2048", "cost_attention_matching_am2048_validation.json", None, None, "cache/compaction/am2048.json"),
    ("C4 AM 1024", "cost_attention_matching_am1024_validation.json", None, None, "cache/compaction/am1024.json"),
    ("C3 cartridge 1024", "cost_cartridge_cart1k_validation.json", None, None, "cache/cartridge/cart1k.json"),
]
lines = ["| condition | prefix tokens | prefix KV (GB) | batch-1 ms/token | batch-1 s / 128 tok | peak GB (batch 1) | one-time GPU-min |", "|---|---|---|---|---|---|---|"]
for name, cost_file, ntok, kv, one in entries:
    L = lat(cost_file)
    if one.startswith("cache/"):
        j = json.loads(Path(one).read_text()) if Path(one).exists() else {}
        ntok, kv = j.get("prefix_tokens_after", j.get("n_tokens")), j.get("prefix_kv_gb")
        one_time = f"{j.get('gpu_minutes_total', j.get('train_log', {}).get('gpu_minutes_total', float('nan'))):.1f}"
    else:
        one_time = "0" if ntok == 0 else "0 (prefill only)"
    f = lambda k, d=1: ("" if L.get(k) is None else f"{L[k]:.{d}f}")
    lines.append(f"| {name} | {ntok if ntok is not None else ''} | {kv if kv is None else f'{kv:.2f}'} | {f('ms_per_token_mean')} | {f('latency_s_mean', 2)} | {f('peak_mem_gb')} | {one_time} |")
(R / "cost_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
