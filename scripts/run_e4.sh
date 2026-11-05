#!/bin/bash
# E4: swap / stack / conflict of deployment policies (CoSApien) as cartridges vs LoRAs vs in context. Qwen3-8B,
# non-thinking. Waits for E3 (needs cart1k + lora_spec_r16) and for the E4 data (data/e4/train_*.jsonl).
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/.hf_home
PY=/venv/compaction/bin/python
G="$PY scripts/13_e4_generate.py"
kill_vllm() { pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 5; }
while ! grep -q "E3 DONE" logs/e3.log; do sleep 60; done
while [ ! -f data/e4/train_film_production.jsonl ]; do sleep 60; done
declare -A S=([game_development]=game [book_publisher_arab]=kidsbooks [film_production]=film)
ALL=game_development,book_publisher_arab,film_production

echo "=== $(date +%T) 1. teacher rollouts (policy in context) on generated train/val prompts"
for P in "${!S[@]}"; do s=${S[$P]}
  for rows in train val; do $G --name teacher_${s}_$rows --rows $rows --policies $P --system policy:$P; kill_vllm; done
done
echo "=== $(date +%T) 2. distil each policy into a cartridge and a LoRA"
for P in "${!S[@]}"; do $PY scripts/12_train_policy_artifacts.py --policy $P || echo "FAILED artifacts $P"; done
echo "=== $(date +%T) 3. LoRA sums"
for P in "${!S[@]}"; do s=${S[$P]}
  [ -d cache/lora/sum_spec_$s ] || $PY scripts/14_merge_lora.py --out cache/lora/sum_spec_$s cache/lora/lora_spec_r16 cache/lora/lora_$s
done
[ -d cache/lora/sum_game_kidsbooks ] || $PY scripts/14_merge_lora.py --out cache/lora/sum_game_kidsbooks cache/lora/lora_game cache/lora/lora_kidsbooks

echo "=== $(date +%T) 4a. SWAP: every single artifact on all 120 CoSApien test prompts"
$G --name base --system none; kill_vllm
$G --name incontext_spec --system spec; kill_vllm
for P in "${!S[@]}"; do s=${S[$P]}
  $G --name incontext_$s --system policy:$P; kill_vllm
  $G --name lora_$s --lora cache/lora/lora_$s; kill_vllm
  $G --name cart_$s --prefixes cache/cartridge/pol_$s.pt
done
$G --name lora_spec --lora cache/lora/lora_spec_r16; kill_vllm
$G --name cart_spec --prefixes cache/cartridge/cart1k.pt

echo "=== $(date +%T) 4b. STACK: Model Spec + policy, on that policy's 40 prompts"
for P in "${!S[@]}"; do s=${S[$P]}
  $G --name stack_incontext_$s --system stack:$P --policies $P; kill_vllm
  $G --name stack_lora_$s --lora cache/lora/sum_spec_$s --policies $P; kill_vllm
  $G --name stack_cart_$s --prefixes cache/cartridge/cart1k.pt,cache/cartridge/pol_$s.pt --policies $P
  $G --name stack_cartrev_$s --prefixes cache/cartridge/pol_$s.pt,cache/cartridge/cart1k.pt --policies $P
done

echo "=== $(date +%T) 4c. CONFLICT: game + kidsbooks together, on both policies' prompts"
GK=game_development,book_publisher_arab
$G --name conflict_incontext --system policy:game_development+book_publisher_arab --policies $GK; kill_vllm
$G --name conflict_incontext_rev --system policy:book_publisher_arab+game_development --policies $GK; kill_vllm
$G --name conflict_lora --lora cache/lora/sum_game_kidsbooks --policies $GK; kill_vllm
$G --name conflict_cart --prefixes cache/cartridge/pol_game.pt,cache/cartridge/pol_kidsbooks.pt --policies $GK
$G --name conflict_cart_rev --prefixes cache/cartridge/pol_kidsbooks.pt,cache/cartridge/pol_game.pt --policies $GK

echo "=== $(date +%T) 4d. SAFETY RETENTION: does unlocking game violence also unlock jailbreaks? (jb val, 140)"
$G --name jb_incontext_game --rows jb --system policy:game_development; kill_vllm
$G --name jb_stack_incontext_game --rows jb --system stack:game_development; kill_vllm
$G --name jb_lora_game --rows jb --lora cache/lora/lora_game; kill_vllm
$G --name jb_stack_lora_game --rows jb --lora cache/lora/sum_spec_game; kill_vllm
$G --name jb_cart_game --rows jb --prefixes cache/cartridge/pol_game.pt
$G --name jb_stack_cart_game --rows jb --prefixes cache/cartridge/cart1k.pt,cache/cartridge/pol_game.pt
echo "=== E4 DONE $(date +%T)"
