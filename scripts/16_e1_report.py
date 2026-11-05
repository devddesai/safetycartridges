"""E1 table + plot: spec effect vs Qwen3 model size, thinking vs non-thinking (validation).

python scripts/16_e1_report.py  -> results/e1_size_sweep.md, results/plot_e1_size_sweep.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sc.metrics import summarize
from sc.rollouts import read_jsonl

R = Path("data/rollouts")
SIZES = ["0.6b", "1.7b", "4b", "8b", "14b", "32b"]
SIZE_B = {"0.6b": 0.6, "1.7b": 1.7, "4b": 4, "8b": 8, "14b": 14, "32b": 32}


def path(cond, size, think, jb):
    if size == "8b":  # session-1 files (no model tag)
        t = "_".join(x for x in [("jb" if jb else ""), ("thinking" if think else "")] if x)
        return R / f"{cond}{'_' + t if t else ''}_validation.jsonl"
    t = "_".join(x for x in [f"q3-{size}", ("jb" if jb else ""), ("thinking" if think else "")] if x)
    return R / f"{cond}_{t}_validation.jsonl"


import re
_W = lambda t: re.findall(r"[a-z0-9']+", t.lower())
_SPEC_8 = None
# 8-grams that also occur in ordinary assistant language are not evidence of copying: exclude those seen in the
# no-spec 8B responses (a reference model that never saw the spec in context).


def _spec_grams():
    global _SPEC_8
    if _SPEC_8 is None:
        w = _W(Path("data/spec/model_spec_plain.txt").read_text())
        g = {tuple(w[i:i + 8]) for i in range(len(w) - 7)}
        for f in ["no_spec_validation", "no_spec_jb_validation"]:
            for r in read_jsonl(R / f"{f}.jsonl"):
                ww = _W(r["response"]); g -= {tuple(ww[i:i + 8]) for i in range(len(ww) - 7)}
        _SPEC_8 = g
    return _SPEC_8


def copy_rate(recs, min_hits=2):
    """Share of responses containing >= min_hits distinct 8-word runs that appear verbatim in the spec."""
    g = _spec_grams()
    def hits(t):
        w = _W(t); return len({tuple(w[i:i + 8]) for i in range(len(w) - 7)} & g)
    return sum(hits(r["response"]) >= min_hits for r in recs) / max(1, len(recs))


def get(cond, size, think, jb):
    p = path(cond, size, think, jb)
    recs = read_jsonl(p)
    if not recs:
        return None
    s = summarize(recs)
    s["spec_copy_rate"] = copy_rate(recs)
    return s if s["n_judged"] >= 0.9 * s["n"] else None


def main():
    rows, data = [], {}
    for think in (False, True):
        for size in SIZES:
            c = {}
            for cond in ("no_spec", "spec_in_context"):
                c[cond] = (get(cond, size, think, True), get(cond, size, think, False))
            data[(think, size)] = c
            f = lambda s, k: f"{s[k]:.2f}" if s and k in s else "–"
            d = lambda a, b, k: f"{b[k] - a[k]:+.2f}" if a and b and k in a and k in b else "–"
            (j1, m1), (j2, m2) = c["no_spec"], c["spec_in_context"]
            rows.append(f"| {'think' if think else 'no-think'} | {size} | {f(j1,'harmful_unsafe_rate')} | {f(j2,'harmful_unsafe_rate')} | "
                        f"**{d(j1,j2,'harmful_unsafe_rate')}** | {f(m1,'harmful_unsafe_rate')} | {f(m2,'harmful_unsafe_rate')} | "
                        f"{f(m1,'benign_false_refusal_rate')} | {f(m2,'benign_false_refusal_rate')} | **{d(m1,m2,'benign_false_refusal_rate')}** | "
                        f"{f(m1,'benign_helpfulness')} | {f(m2,'benign_helpfulness')} | {f(j1,'spec_copy_rate')} | {f(j2,'spec_copy_rate')} | {f(m2,'spec_copy_rate')} |")
    hdr = ("| mode | size | JB unsafe: no spec | JB unsafe: spec | **Δ JB unsafe** | plain unsafe: no spec | plain unsafe: spec | "
           "false refusal: no spec | false refusal: spec | **Δ false refusal** | helpful: no spec | helpful: spec | "
           "spec-copy JB: no spec | spec-copy JB: spec | spec-copy plain: spec |\n" + "|---" * 15 + "|")
    md = "# E1 — spec effect vs Qwen3 size (validation; JB n=140, plain harmful n=140, benign n=110)\n\n" + hdr + "\n" + "\n".join(rows) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/e1_size_sweep.md").write_text(md)
    print(md)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for think, color in ((False, "#4C78A8"), (True, "#E45756")):
        lab = "thinking" if think else "non-thinking"
        for ax, (k, idx, title) in zip(axes, [("harmful_unsafe_rate", 0, "Jailbreak unsafe rate"),
                                              ("benign_false_refusal_rate", 1, "Benign false-refusal rate"),
                                              ("harmful_unsafe_rate", 1, "Plain harmful unsafe rate")]):
            xs, a, b = [], [], []
            for size in SIZES:
                c = data[(think, size)]
                s1, s2 = c["no_spec"][idx], c["spec_in_context"][idx]
                if s1 and s2 and k in s1 and k in s2:
                    xs.append(SIZE_B[size]); a.append(s1[k]); b.append(s2[k])
            ax.plot(xs, a, "--o", color=color, alpha=0.6, label=f"{lab}, no spec")
            ax.plot(xs, b, "-o", color=color, label=f"{lab}, full spec")
            ax.set_xscale("log"); ax.set_xticks([SIZE_B[s] for s in SIZES]); ax.set_xticklabels(SIZES)
            ax.set_title(title); ax.set_xlabel("Qwen3 size"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("results/plot_e1_size_sweep.png", dpi=130)


if __name__ == "__main__":
    main()
