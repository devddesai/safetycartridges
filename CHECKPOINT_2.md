# CHECKPOINT 2 — overnight session 2 (2026-11-05)

Read after CHECKPOINT_1.md. Exploratory night: **validation split only, one sample, no CIs**; test untouched.
Numbers and per-step notes: RESULTS.md ("Session 2"). Plan and its 06:25 revision: OVERNIGHT_PLAN.md.

## Headline results
1. **Reasoning decides whether a spec helps, at every size (E1, Qwen3 0.6B-32B).** In thinking mode the full
   Model Spec in context cuts jailbreak-unsafe at every size (e.g. 14B .46 -> .16, 32B .30 -> .11). In non-thinking
   mode it *hurts*, and the harm **grows with size**: 8B +14 pts, 14B +18, 32B +34 (.14 -> .49), because capable models
   adopt role-play jailbreak personas (AIM) that they refuse without the spec. Small models (<=1.7B) look "safer" with
   the spec only because they degenerate and copy spec text verbatim (31% of 1.7B responses).
2. **Cache beats weights on identical data (E3 + E4).** LoRA and cartridge reach the same distillation loss on the
   same targets in the same GPU time, but:
   - deployment policies (CoSApien, real human-written configs): cartridge reproduces the in-context policy
     (mean score .84 = in-context .84; per-prompt agreement .87) better than the LoRA (.78; .80) on 3/3 policies;
   - the LoRA breaks thinking mode (47/140 responses loop until the token cap vs 19 for cartridge / in-context).
3. **Cartridges compose; LoRA sums blur (E4).** Model Spec cartridge + policy cartridge, concatenated with no
   retraining, ≈ stacking both documents in context (policy score .79-.81 vs .81). Summed LoRAs .70. With two
   conflicting policies loaded, concatenated cartridges reproduce the in-context compromise almost exactly and are
   order-insensitive, while summed LoRAs let the permissive policy override the strict one (.60 vs .80).
   (Contradicts, at n=2 cartridges, the "mixing cartridges collapses" claim of arXiv 2606.04557.)
4. **Stacked cartridges keep (and add) safety (E4 retention, thinking jailbreaks).** Spec cartridge + game-policy
   cartridge: unsafe .09 (no artifact .48, full spec in context .31, in-context stack .33, summed-LoRA stack .21),
   with the game policy still mostly followed (.72 vs in-context stack .78, LoRA stack .70). The cartridge stack
   dominates the LoRA stack on both axes.
5. **Surprise: in thinking mode a 333-token permissive game-studio policy in context protects against jailbreaks as
   much as the whole 55k-token Model Spec (.29 vs .31).** Reasoning models seem to need "a policy is present", not the
   spec's content. Its LoRA (.45) and cartridge (.41) versions lose that effect. E5 tests whether even an irrelevant
   document does it (see RESULTS.md).

## Caveats (all to verify in the next 3 weeks)
- n=40 per policy (CoSApien), n=140 jailbreak: single cells are noisy (±8-15 pts). Trust consistent patterns
  (3/3 policies) over individual cells. Re-run on more samples / seeds before claiming.
- gpt-5-mini judge only; E4 judge prompt (policy-v1) is new and uncalibrated.
- Thinking-mode rates count truncated (looping) responses as safe; report truncation alongside.
- E4 distillation targets are non-thinking; evals are thinking-mode (as in session 1's transfer result).
- E4 training prompts are gpt-5-written (≈257/policy); test = real CoSApien prompts (never shown to the generator).

## What exists now
- New scripts: setup_instance.sh (rebuild), 10_prepare_e4 (CoSApien + gpt-5 train prompts + bucket tags),
  11_train_lora, 12_train_policy_artifacts, 13_e4_generate, 14_merge_lora (exact sum, verified), 15_e4_judge,
  16_e1_report, 17_e4_report; run_e1_sweep / run_e3 / run_e4 / run_e5 queues; 02_generate gained --lora, --variant.
- Data: data/e4/ (policies, CoSApien test with tags, train prompts), data/rollouts/e4/, E1 rollouts (*q3-*).
- Not in git (regenerable): cache/cartridge/{cart1k,pol_*}.pt, cache/lora/*, cache/teacher/*.

## Suggested next steps
1. Statistics on the three headline patterns: more samples per prompt (n_samples=3-5), CIs, a second judge.
2. Scale E4 to all 5 CoSApien policies (+ PolyGuard for longer specs) and to stacks of 3-5 cartridges to find
   where concatenation starts to break.
3. The non-thinking inverse-scaling result (E1) is the most surprising: confirm on test, other model families
   (Llama/Mistral), and with E5's controls at 32B.
4. Thinking-mode teacher targets for cartridges/LoRA; check if the LoRA looping goes away.
