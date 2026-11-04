#!/bin/bash
cd /workspace/safety-cartridges
source /venv/compaction/bin/activate
python scripts/02_generate.py --condition 2 --split validation --limit 50 --tag pilot50_v2 --preamble v2 > logs/gen_c2_val_pilot50_v2.log 2>&1
python scripts/02_generate.py --condition 2 --split validation --limit 50 --tag pilot50_v3 --preamble v3 > logs/gen_c2_val_pilot50_v3.log 2>&1
source /venv/main/bin/activate
python scripts/03_judge.py data/rollouts/spec_in_context_pilot50_v2_validation.jsonl data/rollouts/spec_in_context_pilot50_v3_validation.jsonl > logs/judge_c2_val_pilot50_v2v3.log 2>&1
echo DONE >> logs/judge_c2_val_pilot50_v2v3.log
