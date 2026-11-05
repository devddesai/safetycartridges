#!/bin/bash
# Judge any rollout file with unjudged records, one file at a time (avoids 429s). Loops until STOP file.
cd "$(dirname "$0")/.."
while [ ! -f logs/STOP_JUDGE ]; do
  for f in $(ls -t data/rollouts/*${JUDGE_GLOB:-q3-}*.jsonl 2>/dev/null); do
    n=$(python3 -c "import json,sys; print(sum(1 for l in open('$f') if (lambda j: (not j) or (\"error\" in j))(json.loads(l).get(\"judge\"))))")
    [ "$n" -gt 0 ] && { echo "$(date +%T) judging $f ($n)"; /venv/main/bin/python scripts/03_judge.py $f 2>&1 | tail -2; }
  done
  sleep 120
done
