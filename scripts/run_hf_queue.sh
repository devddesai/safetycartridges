#!/bin/bash
# HF-side queue (runs after the vLLM queue frees the GPU): condition 4 compaction sweep + condition 3 cartridge,
# each evaluated on the validation split and judged in the background.
cd /workspace/safety-cartridges
while [ ! -f logs/vllm_queue.done ]; do sleep 10; done
judge() { (source /venv/main/bin/activate && python scripts/03_judge.py "$@" > "logs/judge_$(basename "$1" .jsonl).log" 2>&1) & }
source /venv/compaction/bin/activate
for T in 2048 1024 4096; do
  python scripts/04_compact.py --target-tokens $T --queries self,train --n-train-convs 96 --tag am${T} > logs/compact_am${T}.log 2>&1
  python scripts/02b_generate_hf.py --condition 4 --prefix cache/compaction/am${T}.pt --split validation --tag am${T} > logs/gen_c4_am${T}_val.log 2>&1
  judge data/rollouts/attention_matching_am${T}_validation.jsonl
done
python scripts/05_train_cartridge.py --n-tokens 1024 --steps 300 --batch 16 --micro-batch 8 --tag cart1k > logs/train_cart1k.log 2>&1
python scripts/02b_generate_hf.py --condition 3 --prefix cache/cartridge/cart1k.pt --split validation --tag cart1k > logs/gen_c3_cart1k_val.log 2>&1
judge data/rollouts/cartridge_cart1k_validation.jsonl
wait
echo HF_QUEUE_DONE > logs/hf_queue.done
