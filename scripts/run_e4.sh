#!/bin/bash
# E4: swap / stack / conflict of deployment policies (CoSApien) as cartridges vs LoRAs vs in context. Qwen3-8B.
# Waits for E3 (needs cart1k + lora_spec_r16) and for the E4 data (data/e4/train_*.jsonl).
#
# Modes (changed 06:20, Dev): distillation targets (teacher rollouts) stay NON-thinking, the same recipe as cart1k
# (session 1: caches trained on non-thinking targets transfer to thinking-mode generation). ALL evaluation generation
# runs in THINKING mode, because E1 / session 1 showed the Model Spec only improves jailbreak safety when the model
# reasons (8B non-thinking: 0.34 -> 0.49 unsafe; thinking: 0.46 -> 0.29). The spec half of a spec+policy stack is
# therefore only measurable with thinking on. To re-run non-thinking later: E4_MODE= E4_SFX=_nothink bash scripts/run_e4.sh
# Judging: the judge watchers only glob *q3-* files, so E4 judges its own rollouts (15_e4_judge.py, resumable) and
# writes results/e4_report.md at the end.
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/.hf_home
PY=/venv/compaction/bin/python
T=${E4_MODE-"--thinking"}
SFX=${E4_SFX-}
G="$PY scripts/13_e4_generate.py"
J="/venv/main/bin/python scripts/15_e4_judge.py"
kill_vllm() { pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 5; }
while ! grep -q "E3 DONE" logs/e3.log 2>/dev/null; do sleep 60; done
while [ ! -f data/e4/train_film_production.jsonl ]; do sleep 60; done
declare -A S=([game_development]=game [book_publisher_arab]=kidsbooks [film_production]=film)
ALL=game_development,book_publisher_arab,film_production
echo "=== $(date +%T) E4 start: eval mode='${T:-non-thinking}' suffix='$SFX'"

echo "=== $(date +%T) 1. teacher rollouts (policy in context, NON-thinking distillation targets) on generated train/val prompts"
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

echo "=== $(date +%T) 4a. SWAP: every single artifact on all 120 CoSApien test prompts ($T)"
$G --name base$SFX --system none $T; kill_vllm
$G --name incontext_spec$SFX --system spec $T; kill_vllm
for P in "${!S[@]}"; do s=${S[$P]}
  $G --name incontext_$s$SFX --system policy:$P $T; kill_vllm
  $G --name lora_$s$SFX --lora cache/lora/lora_$s $T; kill_vllm
  $G --name cart_$s$SFX --prefixes cache/cartridge/pol_$s.pt $T
done
$G --name lora_spec$SFX --lora cache/lora/lora_spec_r16 $T; kill_vllm
$G --name cart_spec$SFX --prefixes cache/cartridge/cart1k.pt $T
$J >> logs/judge_e4.log 2>&1 &

echo "=== $(date +%T) 4b. STACK: Model Spec + policy, on that policy's 40 prompts ($T)"
for P in "${!S[@]}"; do s=${S[$P]}
  $G --name stack_incontext_$s$SFX --system stack:$P --policies $P $T; kill_vllm
  $G --name stack_lora_$s$SFX --lora cache/lora/sum_spec_$s --policies $P $T; kill_vllm
  $G --name stack_cart_$s$SFX --prefixes cache/cartridge/cart1k.pt,cache/cartridge/pol_$s.pt --policies $P $T
  $G --name stack_cartrev_$s$SFX --prefixes cache/cartridge/pol_$s.pt,cache/cartridge/cart1k.pt --policies $P $T
done
$J >> logs/judge_e4.log 2>&1 &

echo "=== $(date +%T) 4c. CONFLICT: game + kidsbooks together, on both policies' prompts ($T)"
GK=game_development,book_publisher_arab
$G --name conflict_incontext$SFX --system policy:game_development+book_publisher_arab --policies $GK $T; kill_vllm
$G --name conflict_incontext_rev$SFX --system policy:book_publisher_arab+game_development --policies $GK $T; kill_vllm
$G --name conflict_lora$SFX --lora cache/lora/sum_game_kidsbooks --policies $GK $T; kill_vllm
$G --name conflict_cart$SFX --prefixes cache/cartridge/pol_game.pt,cache/cartridge/pol_kidsbooks.pt --policies $GK $T
$G --name conflict_cart_rev$SFX --prefixes cache/cartridge/pol_kidsbooks.pt,cache/cartridge/pol_game.pt --policies $GK $T
$J >> logs/judge_e4.log 2>&1 &

echo "=== $(date +%T) 4d. SAFETY RETENTION: does unlocking game violence also unlock jailbreaks? (jb val, 140, $T)"
# Same-mode baselines first (no spec / spec in context / spec cartridge / spec LoRA), so the retention table is
# entirely in one mode rather than mixing session-1 non-thinking numbers with thinking-mode E4 runs.
$G --name jb_base$SFX --rows jb --system none $T; kill_vllm
$G --name jb_incontext_spec$SFX --rows jb --system spec $T; kill_vllm
$G --name jb_lora_spec$SFX --rows jb --lora cache/lora/lora_spec_r16 $T; kill_vllm
$G --name jb_cart_spec$SFX --rows jb --prefixes cache/cartridge/cart1k.pt $T
$G --name jb_incontext_game$SFX --rows jb --system policy:game_development $T; kill_vllm
$G --name jb_stack_incontext_game$SFX --rows jb --system stack:game_development $T; kill_vllm
$G --name jb_lora_game$SFX --rows jb --lora cache/lora/lora_game $T; kill_vllm
$G --name jb_stack_lora_game$SFX --rows jb --lora cache/lora/sum_spec_game $T; kill_vllm
$G --name jb_cart_game$SFX --rows jb --prefixes cache/cartridge/pol_game.pt $T
$G --name jb_stack_cart_game$SFX --rows jb --prefixes cache/cartridge/cart1k.pt,cache/cartridge/pol_game.pt $T

echo "=== $(date +%T) 5. judge everything (resumable) and write results/e4_report.md"
wait
$J >> logs/judge_e4.log 2>&1
/venv/main/bin/python scripts/17_e4_report.py ${SFX:+--suffix $SFX} || echo "FAILED e4 report"
echo "=== E4 DONE $(date +%T)"
