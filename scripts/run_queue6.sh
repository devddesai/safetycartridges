#!/bin/bash
# Sixth queue: apples-to-apples batch-1 latency/memory for C1 and C2 on the HF engine (8 prompts), after queue5.
cd /workspace/safety-cartridges
while [ ! -f logs/queue5.done ]; do sleep 20; done
source /venv/compaction/bin/activate
python scripts/02b_generate_hf.py --condition 2 --split validation --limit 8 --tag hf --batch-size 4 --latency-n 8 > logs/gen_c2_hf_latency.log 2>&1
python scripts/02b_generate_hf.py --condition 1 --split validation --limit 8 --tag hf --batch-size 8 --latency-n 8 > logs/gen_c1_hf_latency.log 2>&1
echo DONE > logs/queue6.done
