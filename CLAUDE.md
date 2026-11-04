# CLAUDE.md — Safety Spec Cartridges

## 1. Context and motivation
This is a research project (advised by Prof. Boaz Barak, Harvard). The core idea:

OpenAI's Deliberative Alignment (Guan et al., 2024) trains models to reason over
an explicit safety specification, using SFT on spec-referencing reasoning followed
by RL with a spec-aware judge. The policy ends up baked into the weights.

An alternative is to keep the safety spec available to the model at inference
time. Putting the full spec (~100 pages, likely 60k+ tokens) in context on every
request is expensive. Cartridges (Eyuboglu et al., 2025, HazyResearch) offer a
middle ground: a small, trained KV cache that compresses a long document. The
paper reports roughly 38x less memory than full in-context at similar quality,
and shows that cartridges can be composed by concatenation without retraining.

## 2. The framing
There are three places to put a safety policy:
- In context (in-context learning): flexible, but expensive at inference.
- In a compressed cache (cartridge or KV compaction): the middle ground.
- In the weights (full fine-tuning, LoRA, or a few test-time gradient steps):
  free at inference, but needs retraining whenever the policy changes.

Important nuance: training a cartridge is NOT much cheaper than LoRA fine-tuning,
since both need forward/backward passes through the full frozen model. And a
cartridge costs slightly MORE at inference than a fine-tuned model. The case for
cartridges is therefore mainly about:
- being much cheaper than the full spec in context, and
- modularity: swapping or stacking domain-specific safety policies (for example
  general + violence, or medical vs. kids' platform) on one frozen base model,
  and updating a policy without touching the base model.

The main competitor on modularity is LoRA adapters, which can already be swapped
per request (vLLM multi-LoRA, S-LoRA). So a sharp long-term research question is:
do safety cartridges stack and swap better than LoRA adapters?

## 3. Research questions (long-term)
For each method, measure:
1. Fidelity: how much of the spec-in-context behaviour is recovered?
2. Generalisation: held-out harmful prompts, jailbreaks, and over-refusal on
   benign-but-risky-looking prompts.
3. Composition: does stacking a general + domain policy work, or do they interfere?
4. Updatability: how cheaply and cleanly can the policy be updated when the
   spec changes?
5. Side effects: is general capability preserved?
6. Cost: training compute, inference memory, and latency.

## 4. Related work to build on
- Deliberative Alignment (Guan et al., 2024, OpenAI).
- Cartridges (Eyuboglu et al., 2025): github.com/HazyResearch/cartridges.
- Fast KV Compaction via Attention Matching (Zweiger, Fu, Guo, Kim; ICML 2026):
  builds compact keys/values that reproduce attention outputs per KV head, with
  no slow end-to-end training. Up to ~50x compression.
  Code: github.com/adamzweiger/compaction.
- qTTT, "Let's (not) just put things in Context: Test-Time Training for
  Long-Context LLMs" (Bansal et al., ICLR 2026, arXiv 2512.13898): a few gradient
  steps on the given context that update ONLY the query projection matrices,
  with the KV cache frozen. Large gains on long-context benchmarks.

## 5. Full experimental plan (all phases)
Conditions:
1. No spec (baseline).
2. Full spec in context (teacher / upper reference).
3. Trained spec cartridge.
4. Attention Matching compaction of the spec.
5. Context distillation: SFT (full or LoRA) on condition-2 outputs, without the
   spec in context.
6. qTTT on the spec.
7. Combinations (for example cartridge + qTTT).
8. Composition: general-spec cartridge + domain cartridge, compared with merged
   or stacked LoRAs.

Models:
- Pilot: Qwen3-8B.
- Main: Qwen3-32B (dense, full attention on every layer, so it suits
  cartridges; its safety tuning wasn't built around OpenAI's spec).
- Avoid: models with linear attention (e.g. Qwen3-Next) or heavy sliding-window
  attention (e.g. Gemma 3), since cartridges need a standard full-attention
  KV cache.
- Side check: gpt-oss-20b, which was trained with deliberative alignment. Does
  having the spec in context still help a model that was already spec-trained?

## 6. TODAY'S PLAN (the only thing to do in this session)
Scope is deliberately small. Do NOT start any later phase, even if time remains.
Instead, use spare time to strengthen the pilot (more eval samples, extra
compression ratios, error analysis).

Hardware/model: single H200 NVL (140GB), Qwen/Qwen3-8B, non-thinking mode.

Spec: OpenAI Model Spec from github.com/openai/model_spec. Convert to plain text
and count tokens with the Qwen tokenizer. If it is over 32k tokens, enable YaRN
rope scaling (factor 4) ONLY for the full-spec-in-context condition. If that
fails, fall back to the core safety sections and document exactly which ones
you used. Record a hash of the exact spec text used.

Conditions today: 1 (no spec), 2 (full spec in context), 3 (cartridge),
4 (Attention Matching). No fine-tuning of any kind today.

### Data splits and saving rollouts (prep for post-training next phase)
- Before running any condition, split all prompt sources (harmful + over-refusal)
  into three fixed, disjoint sets with a fixed random seed:
  - train (~60%): used ONLY to generate training targets for future post-training.
  - validation (~20%): used for tuning choices (cartridge size, training steps,
    compression ratio, judge prompt). Never used as training targets.
  - test (~20%): used ONLY for final reported numbers. Do not tune on it.
- Deduplicate across splits, removing near-duplicate prompts so nothing leaks
  between train and test.
- Save the split assignments to data/splits.json and commit them, so every
  later phase uses exactly the same splits.
- If the prompt sets are small, add prompts from other public harmful and
  over-refusal datasets, and document the sources.
- On the TRAIN split, run condition 2 (full spec in context) and save every
  rollout to data/rollouts/spec_in_context_train.jsonl with: prompt ID, prompt,
  full response, judge score and rationale, model name, sampling settings, spec
  hash, and timestamp.
- Save rollouts for every condition on the validation and test splits, in the
  same format, under data/rollouts/.

### Evals
- Harmful prompts: StrongREJECT or HarmBench (refusal / safe completion rate).
- Over-refusal: XSTest (false refusal rate on benign prompts).
- Judge: OPENAI_API_KEY is in the .env file in the project root. Load it from
  there and use an OpenAI model as judge, grading responses against the
  relevant spec text. If the key is missing or fails, use a local judge and flag
  this as a limitation. Use the same judge and judge prompt for every condition.
- Cost: peak inference GPU memory, latency per request, and one-time
  training/compaction cost in GPU-minutes, for every condition.
- Report all final numbers on the TEST split only.

### Milestones (finish and save each before starting the next)
1. Environment set up, model and spec loaded, token count reported.
2. Prompt datasets collected, deduplicated and split; splits.json committed.
3. Eval harness working on conditions 1 and 2 with ~50 validation prompts.
4. Condition 2 rollouts on the train split saved (future SFT data).
5. Full evals for conditions 1 and 2 on validation and test.
6. Condition 4 (Attention Matching), tuned on validation, evaluated on test.
7. Condition 3 (cartridge). Start with a small cartridge and short run, measure
   memory, then scale up if possible. Tune on validation, evaluate on test.
8. Final comparison table, plots, and a short written summary of findings and
   suggested next steps.

## 7. Working rules
- Run long jobs in the background with logs written to disk.
- Commit to git after every milestone, and push to origin after every commit.
- Never commit .env, API keys, model weights, KV caches or large checkpoints.
- Never print, log or commit API keys.
- Keep RESULTS.md updated after every milestone: what was done, the numbers,
  problems hit, and decisions made (with reasons).
- If blocked for more than ~45 minutes, document it, take the simplest
  workaround, and move on.
- Do not modify base model weights in this session.
- Write clean, reusable code (e.g. a shared eval harness and one config per
  condition), so later phases can plug in easily.

## 8. Notes and lessons (added 2026-11-04 after the Qwen3-8B pilot)
- A small model may not do well on these safety tests when handed a long, detailed spec.
  In the pilot, Qwen3-8B (non-thinking) did not reason over the 55k-token Model Spec; it
  simply became more cautious when the spec was present (benign any-refusal 32% -> 43% on
  test, false full refusals 15.5% -> 19.1%, shorter answers), and the preamble wording alone
  swung false refusals between 28% and 48% on the validation pilot. Its safety was already
  near the ceiling without the spec (0-2% unsafe), so the spec had no room to help on
  safety and could only hurt over-refusal.
- Implications: (a) the compression question (do cartridges / compacted caches reproduce
  the in-context behaviour?) is still answerable at 8B and the pilot harness carries over;
  (b) for behavioural claims we likely need to ramp up to a larger model (Qwen3-32B as
  planned), add a thinking-mode arm (deliberative alignment relies on reasoning over the
  spec, so non-thinking mode may be as big a handicap as size), and use harder evals
  (jailbreaks, subtler dual-use prompts) so there is headroom for the spec to help.
- Feasibility: Qwen3-32B fits one H200 (64 GB weights, ~29 GB full-spec KV cache); vLLM and
  Attention Matching are fine, cartridge training needs gradient checkpointing and ~3-4x
  the time of 8B.
