#!/bin/bash
# Third GPU queue: jailbreak evals for every condition, then the Mistral-Nemo headroom check.
cd /workspace/safety-cartridges
while [ ! -f logs/hf_queue2.done ]; do sleep 15; done
judge() { (source /venv/main/bin/activate && python scripts/03_judge.py "$@" > "logs/judge_$(basename "$1" .jsonl).log" 2>&1) & }
source /venv/compaction/bin/activate
for S in validation test; do
  python scripts/02_generate.py --condition 2 --split $S --prompt-set jailbreak > logs/gen_c2_jb_$S.log 2>&1
  judge data/rollouts/spec_in_context_jb_$S.jsonl
  python scripts/02_generate.py --condition 1 --split $S --prompt-set jailbreak > logs/gen_c1_jb_$S.log 2>&1
  judge data/rollouts/no_spec_jb_$S.jsonl
done
for S in validation test; do
  python scripts/02b_generate_hf.py --condition 4 --prefix cache/compaction/am2048.pt --split $S --tag am2048 --prompt-set jailbreak --latency-n 0 > logs/gen_c4_am2048_jb_$S.log 2>&1
  judge data/rollouts/attention_matching_am2048_jb_$S.jsonl
  if [ -f cache/cartridge/cart1k.pt ]; then
    python scripts/02b_generate_hf.py --condition 3 --prefix cache/cartridge/cart1k.pt --split $S --tag cart1k --prompt-set jailbreak --latency-n 0 > logs/gen_c3_cart1k_jb_$S.log 2>&1
    judge data/rollouts/cartridge_cart1k_jb_$S.jsonl
  fi
done
# Mistral-Nemo-12B headroom check (validation only): main prompts and jailbreak prompts, no spec vs spec in context
NEMO=$(ls -d /workspace/.hf_home/hub/models--*Mistral-Nemo-Instruct-2407 2>/dev/null | head -1 | sed 's#.*models--##; s#--#/#')
if [ -n "$NEMO" ]; then
  for PS in main jailbreak; do
    python scripts/02_generate.py --condition 1 --split validation --prompt-set $PS --model $NEMO --model-tag nemo12b > logs/gen_c1_nemo_${PS}.log 2>&1
    python scripts/02_generate.py --condition 2 --split validation --prompt-set $PS --model $NEMO --model-tag nemo12b > logs/gen_c2_nemo_${PS}.log 2>&1
  done
  judge data/rollouts/no_spec_nemo12b_validation.jsonl
  judge data/rollouts/spec_in_context_nemo12b_validation.jsonl
  judge data/rollouts/no_spec_nemo12b_jb_validation.jsonl
  judge data/rollouts/spec_in_context_nemo12b_jb_validation.jsonl
fi
wait
echo DONE > logs/queue3.done
