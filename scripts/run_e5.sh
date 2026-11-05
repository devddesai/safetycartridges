#!/bin/bash
# E5: why does the in-context spec hurt 8B (non-thinking)? Controls: YaRN alone, preamble alone, spec without
# examples, length-matched irrelevant document. Runs between E1 and E3.
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/.hf_home
PY=/venv/compaction/bin/python
while ! grep -q "E1 DONE" logs/e1_sweep.log; do sleep 60; done
for v in yarn_only preamble_only noexamples longdoc; do
  for ps in jailbreak main; do
    echo "=== $(date +%T) $v $ps"
    $PY scripts/02_generate.py --condition 2 --split validation --prompt-set $ps --variant $v || echo "FAILED $v $ps"
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 5
  done
done
for v in longdoc noexamples; do
  echo "=== $(date +%T) $v jailbreak thinking"
  $PY scripts/02_generate.py --condition 2 --split validation --prompt-set jailbreak --variant $v --thinking || echo "FAILED $v think"
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 5
done
# 32B: the largest non-thinking harm (+0.34 JB unsafe); jailbreak set only
for v in yarn_only preamble_only noexamples longdoc; do
  echo "=== $(date +%T) 32B $v jailbreak"
  $PY scripts/02_generate.py --condition 2 --split validation --prompt-set jailbreak --variant $v \
     --model Qwen/Qwen3-32B --model-tag q3-32b --gpu-mem 0.88 || echo "FAILED 32B $v"
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 5
done
echo "=== E5 DONE $(date +%T)"
