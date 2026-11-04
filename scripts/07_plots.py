"""Plots for the final report (reads results/summary.json written by scripts/06_report.py).
Palette: validated categorical slots (blue, orange, aqua, yellow) with text in neutral ink."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
rows = json.loads((R / "summary.json").read_text())
COND_ORDER = ["no_spec", "spec_in_context", "attention_matching", "cartridge"]
COND_LABEL = {"no_spec": "C1 no spec", "spec_in_context": "C2 full spec in context",
              "attention_matching": "C4 attention matching", "cartridge": "C3 cartridge"}
COLORS = {"no_spec": "#2a78d6", "spec_in_context": "#eb6834", "attention_matching": "#1baf7a", "cartridge": "#eda100"}
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def label(r):
    return COND_LABEL[r["condition"]] + (f" [{r['tag']}]" if r["tag"] else "")


split = sys.argv[1] if len(sys.argv) > 1 else "test"
sel = [r for r in rows if r["split"] == split and not r["tag"].startswith("pilot") and "jb" not in r["tag"].split("_")
       and "nemo" not in r["tag"] and "thinking" not in r["tag"] and r.get("benign_false_refusal_rate") is not None]
sel.sort(key=lambda r: (COND_ORDER.index(r["condition"]), r["tag"]))

# 1. Safety vs over-refusal scatter
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for r in sel:
    x, y = r["benign_false_refusal_rate"], r["harmful_refusal_rate"]
    ax.scatter(x, y, s=70, color=COLORS[r["condition"]], edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.annotate(label(r), (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
ax.set_xlabel("False refusal rate on benign prompts (lower is better)")
ax.set_ylabel("Refusal rate on harmful prompts (higher is better)")
ax.set_title(f"Safety vs over-refusal, {split} split", color=INK, loc="left")
ax.grid(True, color="#e6e5e1", linewidth=0.8, zorder=0)
fig.tight_layout(); fig.savefig(R / f"plot_safety_vs_overrefusal_{split}.png", dpi=160); plt.close(fig)

# 2. Grouped bars of the headline metrics
metrics = [("harmful_refusal_rate", "harmful: refusal"), ("harmful_unsafe_rate", "harmful: unsafe"),
           ("benign_false_refusal_rate", "benign: false refusal"), ("benign_helpfulness", "benign: helpfulness"),
           ("spec_compliance_balanced", "spec compliance (bal.)")]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
w = 0.8 / max(1, len(sel))
for i, r in enumerate(sel):
    xs = [j + i * w for j in range(len(metrics))]
    ys = [r.get(m[0]) or 0 for m in metrics]
    ax.bar(xs, ys, width=w * 0.92, color=COLORS[r["condition"]], label=label(r), zorder=3)
ax.set_xticks([j + 0.4 - w / 2 for j in range(len(metrics))]); ax.set_xticklabels([m[1] for m in metrics])
ax.set_ylim(0, 1.05); ax.grid(True, axis="y", color="#e6e5e1", linewidth=0.8, zorder=0)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_title(f"Headline metrics by condition, {split} split", color=INK, loc="left")
fig.tight_layout(); fig.savefig(R / f"plot_metrics_{split}.png", dpi=160); plt.close(fig)

# 3. Compression sweep on validation (condition 4 tags am<T>, condition 3 tags cart*)
val = [r for r in rows if r["split"] == "validation" and r["condition"] in ("attention_matching", "cartridge")
       and r["cost"].get("prefix_tokens") and "jb" not in r["tag"].split("_") and "nemo" not in r["tag"]]
if val:
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for cond in ("attention_matching", "cartridge"):
        pts = sorted([r for r in val if r["condition"] == cond], key=lambda r: r["cost"]["prefix_tokens"])
        if not pts:
            continue
        xs = [r["cost"]["prefix_tokens"] for r in pts]
        axs[0].plot(xs, [r["benign_false_refusal_rate"] for r in pts], "-o", color=COLORS[cond], label=COND_LABEL[cond], lw=2, ms=6)
        axs[1].plot(xs, [r["harmful_refusal_rate"] for r in pts], "-o", color=COLORS[cond], label=COND_LABEL[cond], lw=2, ms=6)
    ref = {r["condition"]: r for r in rows if r["split"] == "validation" and r["tag"] == ""}
    for ax, m, t in ((axs[0], "benign_false_refusal_rate", "benign false refusal"), (axs[1], "harmful_refusal_rate", "harmful refusal")):
        for c in ("no_spec", "spec_in_context"):
            if c in ref:
                ax.axhline(ref[c][m], color=COLORS[c], ls="--", lw=1.5, label=COND_LABEL[c])
        ax.set_xscale("log"); ax.set_xlabel("prefix tokens (log)"); ax.set_title(t + ", validation", loc="left", color=INK)
        ax.grid(True, color="#e6e5e1", linewidth=0.8)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(R / "plot_compression_sweep_validation.png", dpi=160); plt.close(fig)
print("plots written to results/")

# 4. Jailbreak unsafe rate by condition (validation and test)
jb = [r for r in rows if "jb" in r["tag"].split("_") and "nemo" not in r["tag"] and "thinking" not in r["tag"]]
if jb:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    conds = [c for c in COND_ORDER if any(r["condition"] == c for r in jb)]
    w = 0.38
    for j, sp in enumerate(("validation", "test")):
        for i, c in enumerate(conds):
            rr = [r for r in jb if r["condition"] == c and r["split"] == sp]
            if rr:
                ax.bar(i + (j - 0.5) * w, rr[0]["harmful_unsafe_rate"], width=w * 0.92, color=COLORS[c],
                       alpha=1.0 if sp == "test" else 0.55, zorder=3, label=f"{sp}" if i == 0 else None)
                ax.text(i + (j - 0.5) * w, rr[0]["harmful_unsafe_rate"] + 0.01, f"{rr[0]['harmful_unsafe_rate']:.2f}",
                        ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([COND_LABEL[c] for c in conds], fontsize=8)
    ax.set_ylabel("unsafe rate on jailbreak prompts (lower is better)"); ax.set_ylim(0, 0.65)
    ax.grid(True, axis="y", color="#e6e5e1", linewidth=0.8, zorder=0); ax.legend(frameon=False, fontsize=8, title="split (light = validation)")
    ax.set_title("Jailbreak-wrapped harmful prompts: unsafe rate by condition", color=INK, loc="left")
    fig.tight_layout(); fig.savefig(R / "plot_jailbreak_unsafe.png", dpi=160); plt.close(fig)
