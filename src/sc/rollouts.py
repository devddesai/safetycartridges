"""Rollout record schema and JSONL helpers shared by every condition."""
import json
import datetime as _dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_DIR = ROOT / "data" / "rollouts"

MODEL_NAME = "Qwen/Qwen3-8B"
# Qwen3 recommended non-thinking sampling. Fixed for every condition and engine.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 1024, "seed": 0,
            "enable_thinking": False}

CONDITIONS = {
    1: "no_spec",
    2: "spec_in_context",
    3: "cartridge",
    4: "attention_matching",
}


def rollout_path(condition_name: str, split: str, tag: str = "") -> Path:
    ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    return ROLLOUT_DIR / f"{condition_name}{('_' + tag) if tag else ''}_{split}.jsonl"


def make_record(prompt_row: dict, split: str, condition_name: str, response: str, *,
                engine: str, spec_hash: str | None, extra: dict | None = None) -> dict:
    rec = {
        "prompt_id": prompt_row["id"],
        "prompt": prompt_row["prompt"],
        "source": prompt_row["source"],
        "label": prompt_row["label"],
        "category": prompt_row.get("category"),
        "split": split,
        "condition": condition_name,
        "response": response,
        "model": MODEL_NAME,
        "engine": engine,
        "sampling": dict(SAMPLING),
        "spec_hash": spec_hash,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "judge_score": None,
        "judge_rationale": None,
        "judge": None,
    }
    if extra:
        rec.update(extra)
    return rec


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path, rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
