#!/bin/bash
# Second GPU queue: test-split evals for the chosen condition-3/4 prefixes, then the thinking-mode ablation.
cd /workspace/safety-cartridges
while [ ! -f logs/hf_queue.done ]; do sleep 15; done
judge() { (source /venv/main/bin/activate && python scripts/03_judge.py "$@" > "logs/judge_$(basename "$1" .jsonl).log" 2>&1) & }
source /venv/compaction/bin/activate
python scripts/02b_generate_hf.py --condition 4 --prefix cache/compaction/am2048.pt --split test --tag am2048 > logs/gen_c4_am2048_test.log 2>&1
judge data/rollouts/attention_matching_am2048_test.jsonl
if [ -f cache/cartridge/cart1k.pt ]; then
  python scripts/02b_generate_hf.py --condition 3 --prefix cache/cartridge/cart1k.pt --split test --tag cart1k > logs/gen_c3_cart1k_test.log 2>&1
  judge data/rollouts/cartridge_cart1k_test.jsonl
fi
python scripts/02b_generate_hf.py --condition 4 --prefix cache/compaction/am1024.pt --split test --tag am1024 > logs/gen_c4_am1024_test.log 2>&1
judge data/rollouts/attention_matching_am1024_test.jsonl
# thinking-mode ablation (validation only; conditions 1 and 2)
python scripts/02_generate.py --condition 2 --split validation --thinking > logs/gen_c2_val_thinking.log 2>&1
judge data/rollouts/spec_in_context_thinking_validation.jsonl
python scripts/02_generate.py --condition 1 --split validation --thinking > logs/gen_c1_val_thinking.log 2>&1
judge data/rollouts/no_spec_thinking_validation.jsonl
wait
echo DONE > logs/hf_queue2.done
