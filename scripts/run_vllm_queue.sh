#!/bin/bash
# Sequential vLLM jobs (one model instance at a time). Milestones 4 and 5.
cd /workspace/safety-cartridges
source /venv/compaction/bin/activate
while pgrep -f "02_generate.py --condition 2 --split validation --limit 50" > /dev/null; do sleep 5; done
python scripts/02_generate.py --condition 2 --split train      > logs/gen_c2_train.log 2>&1
python scripts/02_generate.py --condition 2 --split validation > logs/gen_c2_val.log 2>&1
python scripts/02_generate.py --condition 2 --split test       > logs/gen_c2_test.log 2>&1
python scripts/02_generate.py --condition 1 --split validation > logs/gen_c1_val.log 2>&1
python scripts/02_generate.py --condition 1 --split test       > logs/gen_c1_test.log 2>&1
echo QUEUE_DONE > logs/vllm_queue.done
