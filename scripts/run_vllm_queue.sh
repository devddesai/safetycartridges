#!/bin/bash
# Sequential vLLM jobs (one model instance at a time): milestones 4 and 5.
# Usage: scripts/run_vllm_queue.sh <preamble>
PRE=${1:-v1}
cd /workspace/safety-cartridges
judge() { (source /venv/main/bin/activate && python scripts/03_judge.py "$@" > "logs/judge_$(basename "$1" .jsonl).log" 2>&1) & }
source /venv/compaction/bin/activate
python scripts/02_generate.py --condition 2 --split train      --preamble $PRE > logs/gen_c2_train.log 2>&1
judge data/rollouts/spec_in_context_train.jsonl
python scripts/02_generate.py --condition 2 --split validation --preamble $PRE > logs/gen_c2_val.log 2>&1
judge data/rollouts/spec_in_context_validation.jsonl
python scripts/02_generate.py --condition 2 --split test       --preamble $PRE > logs/gen_c2_test.log 2>&1
judge data/rollouts/spec_in_context_test.jsonl
python scripts/02_generate.py --condition 1 --split validation > logs/gen_c1_val.log 2>&1
judge data/rollouts/no_spec_validation.jsonl
python scripts/02_generate.py --condition 1 --split test       > logs/gen_c1_test.log 2>&1
judge data/rollouts/no_spec_test.jsonl
wait
echo QUEUE_DONE > logs/vllm_queue.done
