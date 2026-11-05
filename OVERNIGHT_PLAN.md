# Overnight plan — session 2 (2026-11-05), v2

Goal for tonight (per Dev): **interesting results first, rigor later** (3 weeks remain to verify).
Validation split only; single sample; no CIs required tonight (but report n). Test stays untouched.
Rebuild is done (venv, third_party, Qwen3-8B/32B, Nemo). Instance: $3.72/h.

Four experiments, one per open question. Ordered so the GPU never idles: API-only work
(generating eval items and policies with gpt-5) runs while the GPU does sweeps.

## E1 — Who can actually use a spec? (size x thinking sweep; ~2.5 GPU-h, starts first)
Qwen3 {0.6B, 1.7B, 4B, 8B, 14B, 32B} x {thinking, non-thinking} x {no spec, full spec in context},
on jailbreak-val (140, harmful) + benign-val (110, over-refusal); E2 items appended when ready.
All in-context, so vLLM only (YaRN x4 for the spec arm, as before).
Headline plot: spec effect (Δ unsafe, Δ false refusal) vs model size, one line per mode.
Hypothesis from session 1: the sign flips. Spec hurts small/non-thinking models and helps large/thinking ones.
Where is the crossover?
Stop rule: if a 32B thinking run exceeds ~60 min, cut 32B to the jailbreak set only.

## E2 — Does the model use the spec's *details* or just its *vibe*? (API work + ~1 GPU-h)
A "spec-nuance" eval: about 150 prompts, each built by gpt-5 from one specific Model Spec rule where the
spec prescribes a non-default behaviour (non-safety rules included: formatting, privacy specifics,
dual-use things it should help with, no lecturing, ...). Each item stores the rule ID and the expected behaviour;
the judge scores adherence to that one rule (yes/partial/no).
Conditions at 8B (+32B if time): no spec, full spec, `summary2k` (gpt-5 summary of the spec at the
compressed-cache budget), `preamble_only`, AM-2k, cartridge-1k (+ size sweep from E1 for free).
Interesting either way: if AM/cartridge track the full spec on rule-specific items while summary2k
doesn't, compression preserves *content*. If everything tracks summary2k, the spec acts as a vibe.
Gate: if the full spec beats no spec by <10 pts at both 8B and 32B, items aren't discriminating.
Regenerate harder items once (≤30 min), else drop and note it.

## E3 — Cache vs weights on identical data (8B; ~1.5 GPU-h)
LoRA context distillation (r=16, all linear) on the same 757 C2 train rollouts with the same top-20 KL
objective and a similar GPU-minute budget as the cartridge; regenerate cart1k + am2048 alongside.
Evaluate both: jailbreak-val (thinking + non-thinking), benign-val, E2 nuance eval.
Question: which one carries the spec better, and which generalises better off-distribution (jailbreaks)?
Stop rule: LoRA not training sensibly within 45 min, so drop it and note why.

## E4 — Swap and stack: cartridges vs LoRA adapters (8B; ~2.5 GPU-h)
Two synthetic deployment policies with checkable, partly conflicting rules (gpt-5 drafts, I edit):
P_kids (children's tutoring platform) and P_clinic (clinical-professional tool), about 1-2k tokens each,
plus about 200 targeted prompts per policy (train/val split).
- Teacher = policy in context (vLLM). Gate: it must move targeted behaviour by ≥30 pts vs no policy.
- Distil each policy into a 512-token cartridge and a LoRA (same data, same steps).
- **Swap**: each artifact alone follows its own policy (fidelity vs the teacher).
- **Stack**: Model-Spec cartridge + policy cartridge (KV concatenation, no retraining) vs
  LoRA_spec + LoRA_policy (weight sum). Does each still do its job (policy adherence + jailbreak
  safety from the spec)? Interference = drop vs alone.
- **Conflict**: P_kids + P_clinic together. Which wins, and is it order-dependent for cartridges?
Stop rule: if a single-policy cartridge recovers <50% of the teacher effect after one fix (more steps/size),
report it and finish E4 with LoRA vs in-context only.

Literature hook (docs/LITERATURE.md): "Cartridges at Scale" (2606.04557) reports that mixing independently
trained cartridges collapses, so E4 directly tests that claim for policy cartridges. If our spec+policy stack
works, it is a counterpoint; if it collapses, LoRA-merge vs cartridge-concat is the comparison to report.
If the synthetic policies prove weak, swap in CoSApien configs (real deployment policies with test prompts).

## Stretch (only if all of the above finish)
AM-2k on Qwen3-32B thinking (the E1 point with the largest spec effect); partial refusal-ablation
of Qwen3-8B (in memory only) as a model with headroom.

## Rules for the night
- Validation only; test untouched. Notes in RESULTS.md after each experiment; commit + push.
- Same judge (gpt-5-mini, judge v1) for the safety evals; E2/E4 use a rule-adherence rubric (judge v2-rule),
  identical across conditions within the experiment.
- If something runs >45 min past its box: write down why, take the simplest workaround, move on.
- Morning: CHECKPOINT_2.md, headline results first.
