"""Stack LoRA adapters by exact summation (peft 'cat' combination: delta = sum of the adapters' deltas).

python scripts/14_merge_lora.py --out cache/lora/sum_spec_game cache/lora/lora_spec_r16 cache/lora/lora_game
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from sc.rollouts import MODEL_NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("adapters", nargs="+")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cpu")
    names = [f"a{i}" for i in range(len(args.adapters))]
    m = PeftModel.from_pretrained(base, args.adapters[0], adapter_name=names[0])
    for n, a in zip(names[1:], args.adapters[1:]):
        m.load_adapter(a, adapter_name=n)
    m.add_weighted_adapter(names, [1.0] * len(names), adapter_name="sum", combination_type="cat")
    m.set_adapter("sum")
    # verify on one module: delta(sum) == sum of deltas
    mod = next(mm for nm, mm in m.named_modules() if nm.endswith("layers.10.self_attn.q_proj"))
    d = lambda n: (mod.lora_B[n].weight.float() @ mod.lora_A[n].weight.float()) * mod.scaling[n]
    want, got = sum(d(n) for n in names), d("sum")
    rel = float((want - got).norm() / want.norm())
    print(f"merge check (layer 10 q_proj): relative error {rel:.2e}")
    assert rel < 1e-2, "merged adapter is not the sum of the parts"
    for n in names:
        m.delete_adapter(n)
    m.save_pretrained(args.out + "_tmp")
    # peft saves a non-default adapter under a subdir
    src = Path(args.out + "_tmp") / "sum"
    src = src if src.exists() else Path(args.out + "_tmp")
    src.rename(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
