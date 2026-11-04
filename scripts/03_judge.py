"""Judge every unjudged rollout in a JSONL file in place (resumable).
Usage: python scripts/03_judge.py data/rollouts/no_spec_validation.jsonl [more files...]
"""
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sc.rollouts import read_jsonl, write_jsonl
from sc.judge import judge_many, score_from_judge, JUDGE_MODEL
from sc.metrics import summarize, print_summary

for path in sys.argv[1:]:
    recs = read_jsonl(path)
    todo = [i for i, r in enumerate(recs) if not r.get("judge") or "error" in r["judge"]]
    print(f"{path}: {len(recs)} records, {len(todo)} to judge with {JUDGE_MODEL}")
    if todo:
        res = asyncio.run(judge_many([(recs[i]["prompt"], recs[i]["response"]) for i in todo]))
        for i, j in zip(todo, res):
            recs[i]["judge"] = j
            recs[i]["judge_score"] = score_from_judge(j, recs[i]["label"])
            recs[i]["judge_rationale"] = j.get("rationale")
        write_jsonl(path, recs)
        n_err = sum("error" in r["judge"] for r in recs if r.get("judge"))
        print(f"  done; {n_err} judge errors")
    print_summary(summarize(recs))
