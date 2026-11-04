#!/bin/bash
cd /workspace/safety-cartridges && source /venv/main/bin/activate
for f in no_spec_validation no_spec_test spec_in_context_validation spec_in_context_test spec_in_context_train; do
  python scripts/03_judge.py data/rollouts/$f.jsonl > logs/judge_$f.log 2>&1
done
echo REJUDGE_DONE > logs/rejudge.done
