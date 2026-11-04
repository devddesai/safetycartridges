#!/bin/bash
# Seventh queue: compressed conditions in THINKING mode on the jailbreak validation set (after queue6).
cd /workspace/safety-cartridges
while [ ! -f logs/queue6.done ]; do sleep 20; done
judge() { (source /venv/main/bin/activate && python scripts/03_judge.py "$@" > "logs/judge_$(basename "$1" .jsonl).log" 2>&1) & }
source /venv/compaction/bin/activate
python scripts/02b_generate_hf.py --condition 4 --prefix cache/compaction/am2048.pt --split validation --tag am2048 --prompt-set jailbreak --thinking --latency-n 0 --batch-size 48 > logs/gen_c4_am2048_jb_val_thinking.log 2>&1
judge data/rollouts/attention_matching_am2048_jb_thinking_validation.jsonl
python scripts/02b_generate_hf.py --condition 3 --prefix cache/cartridge/cart1k.pt --split validation --tag cart1k --prompt-set jailbreak --thinking --latency-n 0 --batch-size 48 > logs/gen_c3_cart1k_jb_val_thinking.log 2>&1
judge data/rollouts/cartridge_cart1k_jb_thinking_validation.jsonl
wait
echo DONE > logs/queue7.done
