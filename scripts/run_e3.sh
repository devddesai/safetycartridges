#!/bin/bash
# E3: cartridge vs LoRA on identical data/objective (Qwen3-8B). Waits for the E1 sweep to free the GPU.
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/.hf_home
PY=/venv/compaction/bin/python
while ! grep -q "E1 DONE" logs/e1_sweep.log; do sleep 60; done
echo "=== $(date +%T) cartridge (rebuild teacher targets + cart1k)"
[ -f cache/cartridge/cart1k.pt ] || $PY scripts/05_train_cartridge.py --n-tokens 1024 --steps 300 --batch 16 --tag cart1k
echo "=== $(date +%T) LoRA"
[ -d cache/lora/lora_spec_r16 ] || $PY scripts/11_train_lora.py --tag lora_spec_r16
for args in "--prompt-set jailbreak" "--prompt-set main" "--prompt-set jailbreak --thinking" "--prompt-set main --thinking"; do
  echo "=== $(date +%T) LoRA eval $args"
  $PY scripts/02_generate.py --condition 1 --split validation --lora cache/lora/lora_spec_r16 $args || echo "FAILED lora $args"
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 5
done
echo "=== E3 DONE $(date +%T)"
