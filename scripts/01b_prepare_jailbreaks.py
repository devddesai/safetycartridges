"""Build data/prompts_jailbreak.jsonl: every harmful prompt of validation/test wrapped in one of three
jailbreak templates (rotating deterministically by sorted id)."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sc.data import load_prompts
from sc.jailbreaks import TEMPLATES, apply

names = list(TEMPLATES)
out = []
for split in ("validation", "test"):
    rows = sorted([r for r in load_prompts(split) if r["label"] == "harmful"], key=lambda r: r["id"])
    for i, r in enumerate(rows):
        t = names[i % len(names)]
        out.append({"id": f"{r['id']}_jb_{t}", "base_id": r["id"], "template": t, "split": split,
                    "source": f"{r['source']}+{t}", "label": "harmful", "category": r["category"],
                    "base_prompt": r["prompt"], "prompt": apply(t, r["prompt"])})
p = Path("data/prompts_jailbreak.jsonl")
p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
print(len(out), collections.Counter((r["split"], r["template"]) for r in out))
