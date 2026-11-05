# Overnight plan — session 2 (2026-11-05)

## Where we stand
Session 1 showed that compressed specs (Attention Matching 25x, cartridge 54x) reproduce the
in-context spec's behaviour wherever the spec has an effect (Qwen3-8B thinking + jailbreaks;
Mistral-Nemo-12B). All on validation, n=140 per jailbreak set.

**The weakest point is that nothing rules out "any safety-ish prefix does this."** There is no
control for (a) the v2 preamble alone, (b) a short spec summary at the same ~2k-token budget as the
compressed caches, or (c) a compressed *irrelevant* document. If (b) matches the compressed
caches, compression buys nothing. That control decides the paper's story, so it goes first.

The instance was recycled: venv, third_party, caches and weights are gone. Phase 0 rebuilds them.

## Phases (in order; each has a time box and a stop/pivot rule)

### Phase 0 — rebuild + speed (time box 1.5 h)
- `scripts/setup_instance.sh` (new, idempotent): clones, venv, weights (incl. Qwen3-32B).
- Regenerate am2048, cart1k, nemo12b_am2048 with the session-1 commands. Sanity check: one 50-prompt
  run must match session-1 rates within noise.
- Speed: HF thinking evals took 35-55 min/140 prompts at batch ≤32. Raise the batch to cover the whole
  set (compressed prefixes are ≤0.4 GB). **Stop rule:** if ≥3x faster, adopt it; if not working in
  45 min, keep the old engine and drop the lowest-priority evals below.

### Phase 1 — content controls (THE key question; ~2.5 h)
Settings where the spec has an effect: (A) Qwen3-8B **thinking**, jailbreak set; (B) Nemo-12B, plain + jailbreak.
Jailbreak set grows from 140 to 420/split: every harmful prompt is wrapped in all 3 templates (the old
rotating set stays as a subset). This gives roughly √3 tighter CIs.
New conditions (configs + rollouts like the others):
- `preamble_only`: v2 preamble, no spec.
- `summary2k`: about 2k-token summary of the spec's safety content (written once by gpt-5 from the spec,
  hash recorded), in context. This is the same budget as AM-2k.
- `am2048_irrelevant`: AM compaction of a length-matched non-safety document, same recipe. Tests whether
  "a compressed long prefix" alone does it.
Validation first, with bootstrap 95% CIs on every rate (add to metrics.py).
**Decision gate:**
- Compressed spec beats summary2k/preamble_only by more than the CI → content matters and the thesis
  holds. Confirm on TEST (C1, C2, C3, C4, summary2k) for settings A and B, then Phase 2.
- summary2k ≈ compressed ≈ C2 → on these evals the long spec carries no information beyond a summary.
  Record this as a finding, then **pivot**: build a spec-specific eval where details matter (rules
  from the spec's non-safety sections and its [example] cases, rephrased and held out), and re-test.
  Do not run more jailbreak sweeps.

### Phase 2 — weights vs cache on identical data (~1.5 h)
LoRA context distillation (rank 16, attention+MLP) on the same 757 C2 train rollouts and the same KL
objective as the cartridge, with matched GPU-minutes. Evaluate it like C3 on setting A (thinking jb) and on plain
validation (over-refusal). Then a 2-policy swap test if time allows. This is the core framing comparison
(cache vs weights), and the training data already exists.
**Stop rule:** if LoRA training diverges or takes more than 45 min to get working, document it and move on.

### Phase 3 — scale: Qwen3-32B thinking (~3 h, the remaining time)
C1 vs C2 on jailbreak validation in thinking mode (vLLM, YaRN x4).
**Gate:** proceed only if the spec lowers unsafe by ≥8 points (headroom + effect). Then AM-2k compaction
(memory: 64 GB weights + queries; gradient-free) and its eval. If there's no effect, stop 32B and record it.
Fallback for the remaining time: a self-made partial refusal-direction ablation of Qwen3-8B (in-memory
only, base weights untouched). Pick α on train so the plain unsafe rate is about 40%, then C1 vs C2.
Kill it if the spec can't restore refusals at all (ablation too strong to be informative).

## Global rules for the night
- No tuning on test. No plain-prompt sweeps on already-safe Qwen3-8B. No more preamble variants.
- Every rate gets a 95% bootstrap CI. Don't claim differences inside the CI.
- Commit + push after each phase; RESULTS.md updated; `/workspace` is not a volume.
- If something has eaten more than 45 min past its box, write down why, take the simplest workaround, move on.
- Morning handover: CHECKPOINT_2.md (high-level summary first, then details).
