#!/bin/bash
# Fourth queue: thinking-mode ablation on the jailbreak set and on plain validation (after queue_v2).
cd /workspace/safety-cartridges
while [ ! -f logs/queue_v2.done ]; do sleep 20; done
judge() { (source /venv/main/bin/activate && python scripts/03_judge.py "$@" > "logs/judge_$(basename "$1" .jsonl).log" 2>&1) & }
source /venv/compaction/bin/activate
python scripts/02_generate.py --condition 1 --split validation --prompt-set jailbreak --thinking > logs/gen_c1_jb_val_thinking.log 2>&1
judge data/rollouts/no_spec_jb_thinking_validation.jsonl
python scripts/02_generate.py --condition 2 --split validation --prompt-set jailbreak --thinking > logs/gen_c2_jb_val_thinking.log 2>&1
judge data/rollouts/spec_in_context_jb_thinking_validation.jsonl
python scripts/02_generate.py --condition 2 --split validation --thinking > logs/gen_c2_val_thinking.log 2>&1
judge data/rollouts/spec_in_context_thinking_validation.jsonl
python scripts/02_generate.py --condition 1 --split validation --thinking > logs/gen_c1_val_thinking.log 2>&1
judge data/rollouts/no_spec_thinking_validation.jsonl
wait
echo DONE > logs/queue4.done
