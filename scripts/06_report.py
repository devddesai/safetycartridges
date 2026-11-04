"""Aggregate judged rollouts + cost files into results/summary.json, results/summary.md and plots."""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sc.rollouts import read_jsonl
from sc.metrics import summarize, fidelity

R = Path("results")
rows = []
for p in sorted(glob.glob("data/rollouts/*.jsonl")):
    recs = read_jsonl(p)
    if not recs or not any(r.get("judge") for r in recs):
        continue
    s = summarize(recs)
    name = Path(p).stem
    split = recs[0]["split"]
    cond = recs[0]["condition"]
    tag = name[len(cond) + 1: -len(split) - 1] if name.startswith(cond) else ""
    cost_p = R / f"cost_{name}.json"
    cost = json.loads(cost_p.read_text()) if cost_p.exists() else {}
    ref = read_jsonl(f"data/rollouts/spec_in_context_{split}.jsonl") if split != "train" else []
    fid = fidelity(recs, ref) if ref and not (cond == "spec_in_context" and tag == "") else {}
    rows.append({"file": name, "condition": cond, "tag": tag, "split": split, **s, "fidelity": fid, "cost": cost})
(R / "summary.json").write_text(json.dumps(rows, indent=1))

cols = [("n_judged", "n"), ("harmful_refusal_rate", "harm refusal"), ("harmful_unsafe_rate", "harm unsafe"),
        ("harmful_harm_score", "harm score"), ("benign_false_refusal_rate", "benign full-refusal"),
        ("benign_any_refusal_rate", "benign any-refusal"), ("benign_helpfulness", "benign helpful"),
        ("spec_compliance_balanced", "spec compl. (bal)"), ("mean_output_tokens", "out toks"),
        ("fid_refused_agreement", "agree w/ C2 (refused?)"), ("fid_compliance_mad", "|Δ compl.| vs C2")]
for r in rows:
    for k, v in r["fidelity"].items():
        r["fid_" + k] = v
lines = ["| file | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
for r in sorted(rows, key=lambda r: (r["split"], r["condition"], r["tag"])):
    vals = []
    for k, _ in cols:
        v = r.get(k)
        vals.append("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
    lines.append(f"| {r['file']} | " + " | ".join(vals) + " |")
(R / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
