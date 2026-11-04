"""Prompt sources, near-duplicate removal, and fixed train/val/test splits."""
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TP = ROOT / "third_party"
PROMPTS_JSONL = ROOT / "data" / "prompts.jsonl"
SPLITS_JSON = ROOT / "data" / "splits.json"

SOURCES = {
    "strongreject": "StrongREJECT full dataset (Souly et al. 2024), 313 forbidden prompts, "
                    "github.com/alexandrasouly/strongreject",
    "harmbench": "HarmBench text behaviours, FunctionalCategory=standard only (no context string, no "
                 "copyright), 200 prompts, github.com/centerforaisafety/HarmBench",
    "xstest_unsafe": "XSTest v2 contrast (unsafe) prompts, 200, github.com/paul-rottger/exaggerated-safety",
    "xstest_safe": "XSTest v2 safe prompts (benign, superficially unsafe-looking), 250",
    "orbench_hard": "OR-Bench-Hard-1K (Cui et al. 2024), random subset of 300 benign hard prompts, "
                    "HF bench-llm/or-bench config or-bench-hard-1k",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def _shingles(s: str, n: int = 3) -> set:
    w = _norm(s).split()
    if len(w) < n:
        return {" ".join(w)}
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_sources(seed: int, n_orbench: int = 300) -> list[dict]:
    rows = []
    sr = pd.read_csv(TP / "strongreject/strongreject_dataset/strongreject_dataset.csv")
    for i, r in sr.iterrows():
        rows.append(dict(source="strongreject", label="harmful", category=r["category"],
                         prompt=r["forbidden_prompt"].strip(), orig_id=f"sr_{i}"))
    hb = pd.read_csv(TP / "HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")
    hb = hb[hb.FunctionalCategory == "standard"]
    for _, r in hb.iterrows():
        rows.append(dict(source="harmbench", label="harmful", category=r["SemanticCategory"],
                         prompt=r["Behavior"].strip(), orig_id=f"hb_{r['BehaviorID']}"))
    xs = pd.read_csv(TP / "exaggerated-safety/xstest_prompts.csv")
    for _, r in xs.iterrows():
        src = "xstest_safe" if r["label"] == "safe" else "xstest_unsafe"
        rows.append(dict(source=src, label="benign" if r["label"] == "safe" else "harmful",
                         category=r["type"], prompt=r["prompt"].strip(), orig_id=f"xs_{r['id']}"))
    from datasets import load_dataset
    ob = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    idx = list(range(len(ob)))
    random.Random(seed).shuffle(idx)
    for i in idx[:n_orbench]:
        rows.append(dict(source="orbench_hard", label="benign", category=ob[i]["category"],
                         prompt=ob[i]["prompt"].strip(), orig_id=f"ob_{i}"))
    return rows


def near_dup_groups(rows: list[dict], jaccard_thresh: float = 0.6) -> list[int]:
    """Union-find over near-duplicate pairs (exact normalised match or 3-gram Jaccard >= thresh).
    Returns a group id per row."""
    n = len(rows)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    norms = [_norm(r["prompt"]) for r in rows]
    shs = [_shingles(r["prompt"]) for r in rows]
    for i in range(n):
        for j in range(i):
            if norms[i] == norms[j]:
                parent[find(i)] = find(j)
                continue
            inter = len(shs[i] & shs[j])
            if inter and inter / len(shs[i] | shs[j]) >= jaccard_thresh:
                parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def dedup(rows: list[dict], jaccard_thresh: float = 0.6) -> tuple[list[dict], list[dict]]:
    """Drop near-duplicates that share a label (first occurrence kept). Near-duplicates with
    different labels (e.g. XSTest contrast pairs) are kept, and make_splits keeps each
    near-duplicate group inside a single split so nothing leaks across splits."""
    groups = near_dup_groups(rows, jaccard_thresh)
    kept, dropped, seen = [], [], set()
    for r, g in zip(rows, groups):
        key = (g, r["label"])
        if key in seen:
            dropped.append(r)
        else:
            seen.add(key)
            r["dup_group"] = g
            kept.append(r)
    return kept, dropped


def make_splits(rows: list[dict], seed: int, fracs=(0.6, 0.2, 0.2)) -> dict:
    """Stratified by source, grouped by near-duplicate group (a group never straddles splits).
    Deterministic given the seed."""
    rng = random.Random(seed)
    # Assign each dup group to a split by the group's first row's source stratum.
    group_rows = {}
    for r in rows:
        group_rows.setdefault(r["dup_group"], []).append(r)
    by_src = {}
    for g, rs in group_rows.items():
        by_src.setdefault(rs[0]["source"], []).append(g)
    splits = {"train": [], "validation": [], "test": []}
    for src, gs in sorted(by_src.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        n = len(gs)
        n_tr, n_va = int(round(fracs[0] * n)), int(round(fracs[1] * n))
        for name, chunk in zip(splits, (gs[:n_tr], gs[n_tr:n_tr + n_va], gs[n_tr + n_va:])):
            for g in chunk:
                splits[name] += [r["id"] for r in group_rows[g]]
    for k in splits:
        splits[k] = sorted(splits[k])
    return splits


def load_prompts(split: str | None = None, limit: int | None = None, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in PROMPTS_JSONL.read_text().splitlines()]
    if split is not None:
        ids = set(json.loads(SPLITS_JSON.read_text())["splits"][split])
        rows = [r for r in rows if r["id"] in ids]
    if limit is not None and limit < len(rows):
        # Balanced deterministic subset: interleave harmful/benign.
        rng = random.Random(seed)
        rows = sorted(rows, key=lambda r: r["id"])
        rng.shuffle(rows)
        harm = [r for r in rows if r["label"] == "harmful"]
        ben = [r for r in rows if r["label"] == "benign"]
        out = []
        for i in range(limit):
            pool = harm if (i % 2 == 0 and harm) or not ben else ben
            out.append(pool.pop(0))
        rows = out
    return rows
