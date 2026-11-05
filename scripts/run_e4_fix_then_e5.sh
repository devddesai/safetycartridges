#!/bin/bash
# After E4: re-run jb_cart_spec (OOM'd at batch 96 in thinking mode; now 48), judge it, refresh the E4 report, then E5.
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/.hf_home
while ! grep -q "E4 DONE" logs/e4.log; do sleep 60; done
echo "=== $(date +%T) re-run jb_cart_spec"
/venv/compaction/bin/python scripts/13_e4_generate.py --name jb_cart_spec --rows jb --prefixes cache/cartridge/cart1k.pt --thinking
/venv/main/bin/python scripts/15_e4_judge.py data/rollouts/e4/jb_cart_spec.jsonl
/venv/main/bin/python scripts/17_e4_report.py > /dev/null
echo "=== $(date +%T) fix done; starting E5"
bash scripts/run_e5.sh > logs/e5.log 2>&1
