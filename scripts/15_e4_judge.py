"""Judge E4 rollouts in place (resumable). CoSApien/policy rows -> policy judge (sc.judge_policy) against the
prompt's own policy; jailbreak rows (policy None) -> the standard safety judge (sc.judge, v1).

python scripts/15_e4_judge.py [files...]   (default: every data/rollouts/e4/*.jsonl except teacher_*)
"""
import asyncio, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sc import e4, judge, judge_policy
from sc.rollouts import read_jsonl, write_jsonl


def run(path):
    recs = read_jsonl(path)
    todo = [i for i, r in enumerate(recs) if not r.get("judge") or "error" in r["judge"]]
    if not todo:
        return
    P = e4.policies()
    pol = [i for i in todo if recs[i]["policy"]]
    jb = [i for i in todo if not recs[i]["policy"]]
    if pol:
        res = asyncio.run(judge_policy.judge_many([(P[recs[i]["policy"]], recs[i]["prompt"], recs[i]["response"]) for i in pol]))
        for i, j in zip(pol, res):
            recs[i]["judge"] = j
    if jb:
        res = asyncio.run(judge.judge_many([(recs[i]["prompt"], recs[i]["response"]) for i in jb], progress=False))
        for i, j in zip(jb, res):
            recs[i]["judge"] = j
    write_jsonl(path, recs)
    n_err = sum(1 for r in recs if r.get("judge") and "error" in r["judge"])
    print(f"{path}: judged {len(todo)}, errors {n_err}", flush=True)


if __name__ == "__main__":
    files = sys.argv[1:] or [f for f in sorted(glob.glob("data/rollouts/e4/*.jsonl")) if "/teacher_" not in f]
    for f in files:
        run(f)
