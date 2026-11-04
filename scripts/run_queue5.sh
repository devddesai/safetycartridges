#!/bin/bash
# Fifth queue: Attention-Matching compaction of the spec for Mistral-Nemo-12B, evaluated on validation (plain + jailbreak).
cd /workspace/safety-cartridges
while [ ! -f logs/queue4.done ]; do sleep 20; done
judge() { (source /venv/main/bin/activate && python scripts/03_judge.py "$@" > "logs/judge_$(basename "$1" .jsonl).log" 2>&1) & }
source /venv/compaction/bin/activate
NEMO=mistralai/Mistral-Nemo-Instruct-2407
# train-split condition-2 rollouts for Nemo (query source for compaction; not judged)
python scripts/02_generate.py --condition 2 --split train --model $NEMO --model-tag nemo12b > logs/gen_c2_nemo_train.log 2>&1
python scripts/04_compact.py --target-tokens 2048 --queries self,train --n-train-convs 96 --tag nemo12b_am2048 --model $NEMO --model-tag nemo12b > logs/compact_nemo12b_am2048.log 2>&1
for PS in jailbreak main; do
  python scripts/02b_generate_hf.py --condition 4 --prefix cache/compaction/nemo12b_am2048.pt --split validation --tag nemo12b_am2048 --prompt-set $PS --model $NEMO --latency-n 4 > logs/gen_c4_nemo12b_am2048_${PS}_val.log 2>&1
done
judge data/rollouts/attention_matching_nemo12b_am2048_jb_validation.jsonl
judge data/rollouts/attention_matching_nemo12b_am2048_validation.jsonl
wait
echo DONE > logs/queue5.done
