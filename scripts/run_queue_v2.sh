#!/bin/bash
# Re-prioritised single GPU queue (2026-11-04 18:20): signal-bearing experiments first.
cd /workspace/safety-cartridges
while pgrep -f "[0]2b_generate_hf.py --condition 4 --prefix cache/compaction/am4096" > /dev/null; do sleep 10; done
judge() { (source /venv/main/bin/activate && python scripts/03_judge.py "$@" > "logs/judge_$(basename "$1" .jsonl).log" 2>&1) & }
source /venv/compaction/bin/activate
judge data/rollouts/attention_matching_am4096_validation.jsonl
# 1. Qwen3-8B jailbreak headroom (validation): no spec vs spec
python scripts/02_generate.py --condition 1 --split validation --prompt-set jailbreak > logs/gen_c1_jb_validation.log 2>&1
judge data/rollouts/no_spec_jb_validation.jsonl
python scripts/02_generate.py --condition 2 --split validation --prompt-set jailbreak > logs/gen_c2_jb_validation.log 2>&1
judge data/rollouts/spec_in_context_jb_validation.jsonl
# 2. Mistral-Nemo-12B headroom (validation, main + jailbreak), if the download finished
NEMO=mistralai/Mistral-Nemo-Instruct-2407
if [ -f /workspace/.hf_home/hub/models--mistralai--Mistral-Nemo-Instruct-2407/.download_done ]; then
  for PS in main jailbreak; do
    python scripts/02_generate.py --condition 1 --split validation --prompt-set $PS --model $NEMO --model-tag nemo12b > logs/gen_c1_nemo_${PS}.log 2>&1
    python scripts/02_generate.py --condition 2 --split validation --prompt-set $PS --model $NEMO --model-tag nemo12b > logs/gen_c2_nemo_${PS}.log 2>&1
  done
  judge data/rollouts/no_spec_nemo12b_validation.jsonl
  judge data/rollouts/spec_in_context_nemo12b_validation.jsonl
  judge data/rollouts/no_spec_nemo12b_jb_validation.jsonl
  judge data/rollouts/spec_in_context_nemo12b_jb_validation.jsonl
fi
# 3. Condition 3: cartridge (milestone 7), validation eval
python scripts/05_train_cartridge.py --n-tokens 1024 --steps 300 --batch 16 --micro-batch 8 --tag cart1k > logs/train_cart1k.log 2>&1
python scripts/02b_generate_hf.py --condition 3 --prefix cache/cartridge/cart1k.pt --split validation --tag cart1k > logs/gen_c3_cart1k_val.log 2>&1
judge data/rollouts/cartridge_cart1k_validation.jsonl
# 4. Jailbreak evals for the compressed conditions (validation + test), then the plain test evals
for S in validation test; do
  python scripts/02b_generate_hf.py --condition 4 --prefix cache/compaction/am2048.pt --split $S --tag am2048 --prompt-set jailbreak --latency-n 0 > logs/gen_c4_am2048_jb_$S.log 2>&1
  judge data/rollouts/attention_matching_am2048_jb_$S.jsonl
  if [ -f cache/cartridge/cart1k.pt ]; then
    python scripts/02b_generate_hf.py --condition 3 --prefix cache/cartridge/cart1k.pt --split $S --tag cart1k --prompt-set jailbreak --latency-n 0 > logs/gen_c3_cart1k_jb_$S.log 2>&1
    judge data/rollouts/cartridge_cart1k_jb_$S.jsonl
  fi
done
python scripts/02_generate.py --condition 2 --split test --prompt-set jailbreak > logs/gen_c2_jb_test.log 2>&1
judge data/rollouts/spec_in_context_jb_test.jsonl
python scripts/02_generate.py --condition 1 --split test --prompt-set jailbreak > logs/gen_c1_jb_test.log 2>&1
judge data/rollouts/no_spec_jb_test.jsonl
python scripts/02b_generate_hf.py --condition 4 --prefix cache/compaction/am2048.pt --split test --tag am2048 > logs/gen_c4_am2048_test.log 2>&1
judge data/rollouts/attention_matching_am2048_test.jsonl
if [ -f cache/cartridge/cart1k.pt ]; then
  python scripts/02b_generate_hf.py --condition 3 --prefix cache/cartridge/cart1k.pt --split test --tag cart1k > logs/gen_c3_cart1k_test.log 2>&1
  judge data/rollouts/cartridge_cart1k_test.jsonl
fi
wait
echo DONE > logs/queue_v2.done
