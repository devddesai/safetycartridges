#!/bin/bash
# E1: Qwen3 size x thinking x spec sweep on validation (main 250 + jailbreak 140), vLLM, sequential.
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/.hf_home
PY=/venv/compaction/bin/python
for size in 0.6B 1.7B 4B 14B 32B; do
  m=Qwen/Qwen3-$size; tag=q3-$(echo $size | tr A-Z a-z)
  for think in "" "--thinking"; do
    [ "$size" = 8B ] && [ -z "$think" ] && continue   # 8B non-thinking done in session 1
    for cond in 1 2; do
      for ps in jailbreak main; do
        [ "$size" = 8B ] && [ "$ps" = jailbreak ] && continue   # 8B thinking jb done in session 1
        echo "=== $(date +%T) $size think=$think cond=$cond ps=$ps"
        $PY scripts/02_generate.py --condition $cond --split validation --prompt-set $ps \
           --model $m --model-tag $tag $think --gpu-mem 0.88 || echo "FAILED $size $think $cond $ps"
        pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 5
      done
    done
  done
done
echo "=== E1 DONE $(date +%T)"
