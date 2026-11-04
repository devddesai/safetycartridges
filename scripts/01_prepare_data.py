"""Milestone 2: collect prompt sources, dedup, split 60/20/20 with a fixed seed."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sc.data import load_sources, dedup, make_splits, PROMPTS_JSONL, SPLITS_JSON, SOURCES

SEED = 20261104
rows = load_sources(seed=SEED)
print("raw:", len(rows), collections.Counter(r["source"] for r in rows))
kept, dropped = dedup(rows)
print("after dedup:", len(kept), "dropped:", len(dropped))
for r in dropped[:10]:
    print("  dropped near-dup:", r["source"], r["prompt"][:90])
for i, r in enumerate(kept):
    r["id"] = f"p{i:04d}"
with open(PROMPTS_JSONL, "w") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
splits = make_splits(kept, seed=SEED)
by = {k: collections.Counter(next(x for x in kept if x["id"] == i)["source"] for i in v) for k, v in splits.items()}
meta = {"seed": SEED, "n_total": len(kept), "n_dropped_dedup": len(dropped), "sources": SOURCES,
        "dedup": "near-duplicate groups = exact normalised match or 3-gram Jaccard >= 0.6; same-label duplicates within a group dropped (first kept); each group assigned to a single split",
        "counts": {k: dict(v) for k, v in by.items()},
        "label_counts": {k: dict(collections.Counter(next(x for x in kept if x["id"] == i)["label"] for i in v)) for k, v in splits.items()},
        "splits": splits}
SPLITS_JSON.write_text(json.dumps(meta, indent=1))
print(json.dumps({k: meta[k] for k in ["n_total", "n_dropped_dedup", "counts", "label_counts"]}, indent=1))
