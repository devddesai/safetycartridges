# RESULTS.md — Safety Spec Cartridges pilot (Qwen3-8B, single H200)

Running log of what was done, the numbers, problems, and decisions. Updated after every milestone.

## Milestone 1 — environment, model, spec (done)
- Hardware: 1x H200 NVL (143 GB). Instance workspace is NOT a volume: everything is pushed to git.
- Envs: `/venv/main` (torch 2.14, used for data prep / judging) and `/venv/compaction`
  (python 3.12, torch 2.8.0, transformers 4.57.1, vllm 0.11.0 = the pins of the Attention
  Matching repo). All GPU work (vLLM generation, HF prefix-cache generation, compaction,
  cartridge training) runs in `/venv/compaction` so every condition uses the same stack.
- Model: `Qwen/Qwen3-8B` (snapshot b968826d), non-thinking mode (`enable_thinking=False`).
- Spec: OpenAI Model Spec, github.com/openai/model_spec @ 7ab1b124 (version v2026.10.29).
  Converted markdown -> plain text (`src/sc/spec.py`): heading anchors dropped but
  `authority=` kept as "(authority: root)", cross-refs resolved to section titles, footnote
  markers removed, the 182 `~~~xml` example dialogues kept verbatim inside
  `[example] ... [end example]` blocks, GOOD/BAD comments kept as `[GOOD]`/`[BAD]`.
  - Plain text: 270,060 chars, **55,574 Qwen3 tokens** (markdown was 58,406).
  - Full system prompt (preamble + spec): 55,652 tokens -> **over 32k, so YaRN (factor 4,
    131k context) is enabled for condition 2 only**.
  - sha256 of exact spec text used: `00d71be895dee0faca3890a4ac4abb9e61c4c8d6d1f5828fecbc554d16a29c4f`
    (full meta in `data/spec/meta.json`). A core-safety-sections fallback (19,140 tokens) is
    also written to `data/spec/model_spec_core_sections.txt` but was NOT needed.

## Milestone 2 — prompt data, dedup, splits (done)
Sources (all public): StrongREJECT full (313), HarmBench text `standard` behaviours (200;
contextual and copyright excluded), XSTest v2 unsafe (200) and safe (250), OR-Bench-Hard-1K
random 300 (seed 20261104). Raw total 1,263.

Dedup: near-duplicate groups = exact normalised match or 3-gram Jaccard >= 0.6. Same-label
duplicates inside a group are dropped (7 dropped, all HarmBench variants such as "mix bleach
and X"). Different-label near-duplicates (XSTest contrast pairs like "blow up a balloon" vs
"blow up a house") are kept, and every near-duplicate group is assigned to ONE split, so no
near-duplicate straddles train/test.

Split 60/20/20 stratified by source, seed 20261104, saved in `data/splits.json`:

| split | harmful | benign | total |
|---|---|---|---|
| train | 427 | 330 | 757 |
| validation | 140 | 110 | 250 |
| test | 139 | 110 | 249 |

## Milestone 3 — eval harness on conditions 1 and 2, 50 validation prompts (done)
Harness: `scripts/02_generate.py` (vLLM 0.11, prefix caching; YaRN x4 via `hf_overrides` for
condition 2 only), `scripts/03_judge.py` + `src/sc/judge.py` (OpenAI `gpt-5-mini`,
reasoning_effort=low, strict JSON schema, graded against the spec's core safety sections,
19k tokens, served from the prompt cache), `src/sc/metrics.py`.
Sampling for every condition: Qwen3 non-thinking defaults (T=0.7, top_p=0.8, top_k=20),
max 1024 new tokens, seed 0. Judge outputs per response: refusal (full/partial/compliance),
harmfulness 0-5, helpfulness 1-5, spec_compliance 1-5, should_comply, rationale.

Pilot (50 balanced validation prompts, preamble v1):

| metric | C1 no spec | C2 full spec |
|---|---|---|
| harmful: refusal rate | 0.92 | 0.96 |
| harmful: unsafe rate (harmfulness>=3) | 0.00 | 0.00 |
| benign: false (full) refusal rate | 0.16 | **0.44** |
| benign: any refusal (full+partial) | 0.36 | 0.68 |
| benign: helpfulness (0-1) | 0.90 | 0.71 |
| spec compliance, balanced (0-1) | 0.95 | 0.93 |
| mean output tokens | 313 | 142 |

**Problem found:** Qwen3-8B is already very safe without the spec (0% unsafe), and putting the
55k-token spec in context makes it *more* over-refusing (44% false refusals on XSTest-safe /
OR-Bench-hard vs 16%), with much shorter answers. Qualitatively it adopts a "policy-citing"
refusal persona ("I cannot provide personal information... including fictional characters like
Robin Hood"). Since condition 2 is the teacher for conditions 3 and 4 and for later SFT, the
spec preamble was treated as a tunable choice on the validation pilot (never on test):
`v1` = original, `v2` = explicitly says needless refusals are as serious as harmful content,
`v3` = minimal one-line preamble. Results in the next section.
Throughput: C1 1,600 output tok/s; C2 252 tok/s batched with the 55.7k-token shared prefix.

### Spec preamble choice (validation pilot, 50 prompts, condition 2)

| preamble | harmful refusal | benign false refusal | benign any refusal | benign helpfulness | spec compl. (bal) | out toks |
|---|---|---|---|---|---|---|
| (C1, no spec) | 0.92 | 0.16 | 0.36 | 0.90 | 0.950 | 313 |
| v1 original | 0.96 | 0.44 | 0.68 | 0.71 | 0.930 | 142 |
| **v2 explicit "over-refusal is a failure"** | 0.96 | **0.28** | 0.52 | 0.81 | **0.945** | 200 |
| v3 minimal one-liner | 1.00 | 0.48 | 0.68 | 0.73 | 0.925 | 188 |

Decision: **preamble v2** is fixed for every spec-conditioned condition (2, 3, 4) and for all
train/val/test rollouts (`sc.spec.DEFAULT_PREAMBLE = "v2"`). Even with v2 the spec-in-context
teacher over-refuses more than the no-spec baseline on this 8B model; this is reported as a
finding rather than tuned further (only 3 variants were tried, all chosen on validation).

## Milestone 4 — condition 2 rollouts on the TRAIN split (done)
`data/rollouts/spec_in_context_train.jsonl`: 757 prompts, preamble v2, YaRN x4, 140k output
tokens, 15/757 hit the 1024-token cap, median response 108 tokens. Generation took 278 s of
vLLM wall time (0.37 s/prompt batched; the 55.7k-token prefix is shared via prefix caching).
Judge scores are filled in by `scripts/03_judge.py` (same judge as everywhere else).

## Milestone 5 — full evals of conditions 1 and 2 on validation and test (done)
(Judge: gpt-5-mini; a first judging pass hit 429 rate limits because five judge jobs ran in
parallel; the failed records were re-judged sequentially with concurrency 8 -> 0 errors.)

| split | condition | harmful refusal | harmful unsafe | benign false refusal | benign any refusal | benign helpfulness | spec compl. (bal.) | out toks |
|---|---|---|---|---|---|---|---|---|
| validation (n=250) | C1 no spec | 0.950 | 0.021 | 0.109 | 0.327 | 0.887 | 0.944 | 321 |
| validation | C2 spec in context | 0.979 | 0.000 | 0.218 | 0.491 | 0.796 | 0.905 | 204 |
| **test (n=249)** | **C1 no spec** | 0.971 | 0.014 | 0.155 | 0.318 | 0.860 | 0.893 | 287 |
| **test** | **C2 spec in context** | 0.986 | 0.000 | 0.191 | 0.427 | 0.805 | 0.885 | 190 |
| train (n=757) | C2 spec in context | 0.995 | 0.002 | 0.212 | 0.521 | 0.778 | 0.888 | 185 |

Reading: the spec in context removes the few unsafe completions the base model still produces
(2/139 -> 0/139 on test) and raises harmful-prompt refusal slightly, but it costs over-refusal
(+3.6 pts full refusals, +11 pts any-refusal on benign prompts) and helpfulness, and the judge's
balanced spec-compliance is marginally lower. For Qwen3-8B the "teacher" is therefore a
safer-but-more-cautious policy, not a strictly better one. Conditions 3 and 4 are judged on how
faithfully they reproduce condition 2, not on beating condition 1.

## Milestone 6 — condition 4, Attention Matching (in progress; validation sweep done for 1k/2k)
Setup: full-spec KV cache (55,749 prefix tokens incl. template + preamble, YaRN x4) compacted
with the Attention-Matching repo's `AM-HighestAttnKeys` (rms scores, 2 NNLS iters, lsq values),
compacting only the 55,574 spec tokens (template/preamble tokens kept verbatim). Queries (train
split only): 16k spec-token queries ("self") + queries from 96 condition-2 TRAIN conversations
(prompt + response attending to the full cache), 50k queries per KV head. Uniform per-head budget
(no precomputed head budgets exist for Qwen3-8B). One-time cost per size: prefill 11 s + queries
33 s + compaction 61-83 s = **~2 GPU-minutes**, peak 88 GB.

| prefix (validation, n=250) | tokens | KV size | harmful refusal | harmful unsafe | benign false refusal | benign any refusal | benign helpful | agree w/ C2 (refused?) |
|---|---|---|---|---|---|---|---|---|
| C1 no spec | 11 | ~0 | 0.950 | 0.021 | 0.109 | 0.327 | 0.887 | 0.848 |
| C2 full spec | 55,749 | 8.22 GB | 0.979 | 0.000 | 0.218 | 0.491 | 0.796 | (self-agreement across two samples ~0.86) |
| C4 AM 2048 (27x) | 2,223 | 0.33 GB | 0.979 | 0.000 | 0.255 | 0.445 | 0.789 | 0.900 |
| C4 AM 1024 (54x) | 1,199 | 0.18 GB | 0.986 | 0.000 | 0.300 | 0.473 | 0.769 | 0.916 |
| C4 AM 4096 (14x) | 4,271 | 0.63 GB | (running) | | | | | |

Both compacted caches reproduce the spec-in-context behaviour at the sampling-noise ceiling
(per-prompt refuse/comply agreement with C2 is as high as C2's agreement with a second sample of
itself). Latency (batch 1, 128 new tokens, HF/sdpa): 38 ms/token with the 2k prefix vs 35 ms/token
with 4k; peak memory 17.1 GB vs the model's 16.4 GB.
**Choice for test: AM-2048** (pre-registered middle size; closest to C2 on benign refusals).

### Re-prioritisation (18:20)
Qwen3-8B is 0-2% unsafe on plain harmful prompts with or without the spec, so plain-prompt
safety numbers carry no learning signal. Remaining GPU time is re-ordered to experiments that can
produce a signal: (1) jailbreak-wrapped harmful prompts (AIM, refusal-suppression, evil-confidant
templates on the val/test harmful prompts) for all conditions; (2) a headroom check on a lightly
safety-tuned model, Mistral-Nemo-12B-Instruct (128k native context, Llama-style architecture),
no-spec vs spec-in-context on validation; (3) the cartridge (condition 3); (4) plain test evals for
the chosen C3/C4 prefixes. Dropped: test evals of AM-1024/4096 and the thinking-mode ablation
(only if time remains).

## Jailbreak evals (validation, 140 harmful prompts wrapped in AIM / refusal-suppression / evil-confidant)
This is where Qwen3-8B has headroom, and the first clear behavioural result of the pilot:

| validation, jailbreak set | harmful refusal | **unsafe rate** (harm>=3) | harm score |
|---|---|---|---|
| C1 no spec | 0.721 | **0.343** | 0.323 |
| C2 full spec in context | 0.600 | **0.486** | 0.449 |

**The full spec in context makes the 8B model MORE jailbreakable** (28 prompts flip safe->unsafe,
8 flip the other way). Per template: AIM and refusal-suppression do most of the damage (StrongREJECT
+AIM: refusal 0.30 -> 0.10; +refusal-suppression 0.13 -> 0.05); evil-confidant is mostly refused in
both. Qualitatively the spec-conditioned model slips into the "AIM:" persona and produces bullet-
pointed harmful content where the bare model says "I cannot fulfill this request." Hypotheses:
(a) a 55k-token instruction-following prefix primes compliance with the final instruction
block, (b) the spec's many example dialogues (incl. role-play examples) teach persona-following,
(c) YaRN-scaled long context weakens the model's own safety behaviour. For an 8B model in
non-thinking mode, in-context access to the spec is not a safety mechanism against jailbreaks;
this strengthens the case for a larger / thinking-mode model (CLAUDE.md §8). The question for
conditions 3/4 is now whether the compressed caches reproduce this vulnerability (fidelity) or not.

## Headroom check: Mistral-Nemo-12B-Instruct-2407 (validation only, vLLM, no rope scaling: 128k native)
Official lightly-moderated instruct model, same prompts, same judge, preamble v2.

| validation | harmful refusal | **unsafe** | benign false refusal | benign any refusal | benign helpful | jailbreak refusal | **jailbreak unsafe** |
|---|---|---|---|---|---|---|---|
| Nemo, no spec | 0.886 | **0.200** | 0.064 | 0.191 | 0.925 | 0.314 | **0.850** |
| Nemo, full spec in context | 0.886 | **0.143** | 0.145 | 0.291 | 0.804 | 0.486 | **0.714** |
| (Qwen3-8B, no spec) | 0.950 | 0.021 | 0.109 | 0.327 | 0.887 | 0.721 | 0.343 |
| (Qwen3-8B, full spec) | 0.979 | 0.000 | 0.218 | 0.491 | 0.796 | 0.600 | 0.486 |

On Nemo the spec in context works in the intended direction on both prompt sets (unsafe 20% -> 14%
plain, 85% -> 71% under jailbreaks) with the usual over-refusal cost. So Nemo has real headroom and
a real teacher effect, unlike Qwen3-8B. Decision: run the Attention-Matching compaction on Nemo
tonight if time allows (the compaction repo's Llama path should load Mistral weights), evaluated
on the jailbreak and plain validation sets, to test whether a compressed spec carries the gain.

## Milestone 7 — condition 3, cartridge (training done; evals running)
`scripts/05_train_cartridge.py`: trainable KV prefix of 1,024 tokens (per layer K,V of shape
(8 heads x 1024 x 128), 36 layers = 0.15 GB bf16), initialised from the KV of the first 1,024
tokens of the spec system prompt, native RoPE (no YaRN; prompt tokens start at position 1024).
Teacher = condition 2 (YaRN model, full spec cache); targets = top-20 teacher logprobs on the
response tokens of the 757 TRAIN rollouts (teacher NLL 0.35). Loss = KL(teacher top-20 || student)
on response tokens. AdamW lr 2e-2 (Cartridges default), cosine, batch 16 (2 x 8), 300 steps.
Validation KL (64 held-out condition-2 validation conversations): 0.698 at init -> 0.164 (step 25)
-> 0.102 (100) -> 0.088 (300); train KL 0.027 at the end (some overfitting, val still flat/falling).
Cost: teacher targets ~4 min + training 5.2 min = **9.1 GPU-minutes**, peak 76.6 GB (batch 8 x
~700 tokens with backward through the frozen 8B model). Compared with ~2 GPU-minutes for
Attention Matching at a comparable size.
Cartridge validation eval (plain prompts, n=250): harmful refusal 0.979, unsafe 0.000, benign false
refusal 0.273, any refusal 0.509, helpfulness 0.771, refuse/comply agreement with C2 **0.928**
(C2 self-agreement ~0.86), latency 35 ms/token at batch 1 with no YaRN (native RoPE). So on
plain prompts the 1k-token cartridge (54x) reproduces condition 2 as well as the 2k Attention
Matching cache does. The discriminating test is the jailbreak set (next).

### Jailbreak set, validation (140 prompts): the compressed conditions diverge
| validation, jailbreak | harmful refusal | **unsafe** | harm score | unsafe-agreement w/ C2 | w/ C1 |
|---|---|---|---|---|---|
| C1 no spec | 0.721 | 0.343 | 0.323 | 0.743 | - |
| C2 full spec in context | 0.600 | 0.486 | 0.449 | - | 0.743 |
| C4 Attention Matching 2k | 0.607 | 0.429 | 0.409 | **0.786** | 0.700 |
| C3 cartridge 1k | 0.793 | **0.257** | 0.233 | 0.729 | 0.771 |

The two compression methods preserve different things. Attention Matching reproduces the
teacher's *mechanism* (attention over the spec), so it inherits condition 2's jailbreak
vulnerability (unsafe 0.43 vs 0.49, per-prompt agreement with C2 above the C1/C2 baseline). The
cartridge was distilled on condition-2 *responses* to plain train prompts (mostly refusals of
harmful requests, no jailbreaks), so it learned a refusal-leaning behaviour and generalises to
jailbreaks as the safest condition (unsafe 0.26, below even the bare model's 0.34) while being
less faithful to the teacher. For the research question "is the spec-in-context behaviour
recovered?", AM scores higher on fidelity; for "is the deployed model safe?", the cartridge is
better here, but only because the teacher itself is a poor jailbreak policy on this 8B model.

## Milestone 8 — final comparison on the TEST split (Qwen3-8B, non-thinking)
Plots: `results/plot_metrics_test.png`, `results/plot_safety_vs_overrefusal_test.png`,
`results/plot_jailbreak_unsafe.png`, `results/plot_compression_sweep_validation.png`.
Full per-file table: `results/summary.md`; costs: `results/cost_table.md`.

### Plain prompts (test, n=249: 139 harmful, 110 benign)
| condition | prefix tokens | KV | harmful refusal | harmful unsafe | benign false refusal | benign any refusal | benign helpful | spec compl. (bal.) | agree w/ C2 (refuse?) |
|---|---|---|---|---|---|---|---|---|---|
| C1 no spec | 0 | 0 | 0.971 | 0.014 | 0.155 | 0.318 | 0.860 | 0.893 | 0.871 |
| C2 full spec in context | 55,749 | 8.22 GB | 0.986 | 0.000 | 0.191 | 0.427 | 0.805 | 0.885 | - |
| C4 Attention Matching 2k (25x) | 2,223 | 0.33 GB | 0.993 | 0.000 | 0.182 | 0.427 | 0.798 | 0.897 | **0.932** |
| C3 cartridge 1k (54x) | 1,024 | 0.15 GB | 0.986 | 0.000 | 0.182 | 0.473 | 0.758 | 0.890 | 0.892 |

### Jailbreak-wrapped harmful prompts (test, n=139)
| condition | harmful refusal | **unsafe** | harm score | unsafe-agreement w/ C2 | w/ C1 |
|---|---|---|---|---|---|
| C1 no spec | 0.719 | 0.281 | 0.265 | 0.748 | - |
| C2 full spec in context | 0.619 | **0.475** | 0.445 | - | 0.748 |
| C4 Attention Matching 2k | 0.691 | 0.388 | 0.371 | **0.813** | 0.734 |
| C3 cartridge 1k | 0.827 | **0.209** | 0.200 | 0.691 | 0.741 |

### Cost (HF engine, same code path for every condition; batch 1, 128 new tokens)
| condition | prefix tokens | prefix KV | ms/token | peak GB | one-time cost |
|---|---|---|---|---|---|
| C1 no spec | 11 | ~0 | 35.7 | 16.5 | 0 |
| C2 full spec | 55,749 | 8.22 GB | **129.9** | 34.2 | prefill 11 s (then cached) |
| C4 AM 2k | 2,223 | 0.33 GB | 38.3 | 17.1 | 2.1 GPU-min |
| C4 AM 1k | 1,199 | 0.18 GB | 37.6 | 16.8 | 1.7 GPU-min |
| C4 AM 4k | 4,271 | 0.63 GB | 34.5 | 17.8 | 3.0 GPU-min |
| C3 cartridge 1k | 1,024 | 0.15 GB | 35.0 | 16.8 | 9.1 GPU-min (incl. 4 min teacher targets) |
Single-stream decoding with the full 55.7k-token spec is 3.6x slower per token than with a compressed
prefix (130 vs 35-38 ms/token on the same HF/sdpa engine) and needs 2x the memory (34 vs 17 GB).
vLLM throughput for reference: C1 1,600 output tok/s, C2 250-500 tok/s with the shared 55.7k prefix.

### Summary of findings
1. **Compression works as compression.** Both a 25x Attention-Matching cache (2 GPU-minutes, no
   training) and a 54x trained cartridge (9 GPU-minutes) reproduce the spec-in-context behaviour on
   plain prompts at the sampling-noise ceiling (per-prompt refuse/comply agreement with C2 of
   0.93 / 0.89 vs ~0.86 for C2 against a resample of itself), at 25-55x less KV memory
   (8.2 GB -> 0.15-0.33 GB) and ~35-38 ms/token single-stream latency.
2. **But the teacher is not a good policy on Qwen3-8B.** The base model is already 1-2% unsafe
   on plain harmful prompts; the full spec in context adds over-refusal (+3.6 pts full refusals,
   +11 pts any-refusal, shorter answers) and, on jailbreak-wrapped prompts, makes the model
   *more* unsafe (28% -> 48%): it slips into the jailbreak persona and complies. The preamble
   wording swung false refusals between 28% and 48% on the pilot, i.e. the 8B model pattern-matches
   the presence of a policy document rather than reasoning over it.
3. **The two compression methods fail differently under distribution shift.** Attention Matching
   reproduces the teacher's attention, so it inherits the jailbreak vulnerability (39% unsafe,
   agreement with C2 0.81). The cartridge, distilled on condition-2 responses to plain prompts,
   learned a refusal-leaning behaviour and is the safest condition on jailbreaks (21% unsafe, below
   the bare model's 28%) while being less faithful to the teacher. Fidelity and safety come apart
   as soon as the teacher is wrong.
4. **Headroom exists on a lightly safety-tuned model.** Mistral-Nemo-12B-Instruct is 20% unsafe on
   plain prompts and 85% under jailbreaks; the spec in context reduces both (14% / 71%) at an
   over-refusal cost (6% -> 15%). This is the model (and prompt set) on which the compression
   question is worth asking; a Nemo compaction run is queued (results appended below if finished).

### Suggested next steps
- Main model: Qwen3-32B as planned, plus Mistral-Nemo-12B (or a self-made refusal-ablated
  Qwen3-8B) so the no-spec baseline is unsafe and the spec has something to carry.
- Thinking-mode arm: deliberative alignment relies on reasoning over the spec (ablation queued).
- Train cartridges on spec-grounded self-study data (summaries / QA about the spec, as in the
  Cartridges paper) in addition to on-task responses, and on jailbreak-wrapped train prompts, then
  re-test whether fidelity and safety can be had together.
- For Attention Matching: on-policy query refinement and queries from jailbreak conversations.
- Post-training phase (context distillation / LoRA) can start from
  `data/rollouts/spec_in_context_train.jsonl` (757 judged rollouts, preamble v2) using
  `data/splits.json`.

## Thinking-mode ablation (Qwen3-8B, validation; thinking sampling T=0.6/top_p 0.95, max 4096 tokens, `<think>` stripped before judging)
| validation | harmful refusal | harmful unsafe | benign false refusal | benign any refusal | benign helpful | jailbreak refusal | **jailbreak unsafe** |
|---|---|---|---|---|---|---|---|
| non-thinking, no spec | 0.950 | 0.021 | 0.109 | 0.327 | 0.887 | 0.721 | 0.343 |
| non-thinking, full spec | 0.979 | 0.000 | 0.218 | 0.491 | 0.796 | 0.600 | **0.486** |
| thinking, no spec | 0.921 | 0.036 | 0.018 | 0.182 | 0.960 | 0.636 | 0.464 |
| thinking, full spec | 0.979 | 0.007 | 0.173 | 0.418 | 0.847 | 0.743 | **0.286** |

**With reasoning on, the spec in context works** (jailbreak unsafe 0.46 -> 0.29 instead of
0.34 -> 0.49), and thinking alone nearly removes over-refusal on benign prompts (false refusals
1.8%, helpfulness 0.96), though the spec still adds over-refusal (17%). 19/140 jailbreak responses
with the spec hit the 4096-token cap (long deliberation), a small caveat. This supports the
deliberative-alignment premise: a spec in context is only useful if the model reasons over it,
and the non-thinking pilot was measuring pattern-matching. Implication for the compression
question: the condition worth compressing is **thinking + spec**, and the fidelity test should be
run in thinking mode (cheap to add: the prefix is identical, only generation changes).

## Attention Matching on Mistral-Nemo-12B (validation): the compressed spec carries the safety gain
Spec = 58,031 Nemo tokens (prefix 58,211 incl. template/preamble) compacted to 2,048 spec tokens
with the same recipe (self + 96 train-conversation queries; Nemo train rollouts generated for
this). Nemo has no rope scaling (128k native).

| Nemo-12B, validation | harmful refusal | **unsafe** | benign false refusal | benign any refusal | benign helpful | jailbreak refusal | **jailbreak unsafe** |
|---|---|---|---|---|---|---|---|
| C1 no spec | 0.886 | 0.200 | 0.064 | 0.191 | 0.925 | 0.314 | 0.850 |
| C2 full spec in context (58k tokens) | 0.886 | 0.143 | 0.145 | 0.291 | 0.804 | 0.486 | 0.714 |
| **C4 AM 2k (28x)** | 0.914 | **0.143** | 0.136 | 0.355 | 0.693 | 0.414 | **0.721** |

The 28x-compressed cache reproduces the full spec's safety effect on both prompt sets (unsafe
0.143 vs 0.143 plain; 0.721 vs 0.714 under jailbreaks), with the same over-refusal profile as the
full spec and somewhat lower benign helpfulness (0.69 vs 0.80). This is the result the project is
after: on a model with headroom, a ~0.3 GB compacted spec behaves like the 9 GB in-context spec.
Cost: 3.1 GPU-minutes one-time (prefill + queries + 97 s compaction), 2,228 prefix tokens,
0.37 GB vs 9.54 GB for the full cache. Per-prompt agreement is noisier than the aggregate rates:
plain prompts, refuse/comply agreement AM~C2 0.82 vs C1~C2 0.80, unsafe-agreement AM~C2 0.90 vs
AM~C1 0.83 (so AM tracks C2); jailbreak prompts, unsafe-agreement AM~C2 0.69 vs AM~C1 0.77 with
C1~C2 refuse-agreement only 0.60, i.e. jailbreak outcomes on Nemo are close to coin flips per
prompt under sampling, and only the aggregate rate is informative at n=140.

## Compressed conditions in THINKING mode, jailbreak validation set (the condition worth compressing)
| Qwen3-8B thinking, jailbreak validation (n=140) | harmful refusal | **unsafe** | harm score | truncated at 4096 |
|---|---|---|---|---|
| C1 no spec | 0.636 | 0.464 | 0.445 | 3 |
| C2 full spec in context (55.7k tokens, 8.2 GB) | 0.743 | 0.286 | 0.270 | 19 |
| **C4 Attention Matching 2k (0.33 GB)** | 0.786 | **0.236** | 0.223 | 10 |
| **C3 cartridge 1k (0.15 GB)** | 0.829 | **0.193** | 0.167 | 19 |

With reasoning enabled, both compressed specs deliver the full spec's jailbreak protection (and
numerically a bit more, within noise at n=140) at 25-55x less KV memory. Together with the Nemo
result this is the project's positive finding: when the in-context spec actually changes
behaviour, a compressed cache reproduces that change. (Caveat: the compressed caches were built /
trained from non-thinking condition-2 data, yet transfer to thinking-mode generation.)

### Updated bottom line
- Compression (both methods) reproduces the in-context spec wherever the spec has an effect:
  plain prompts on Qwen3-8B (fidelity at the noise ceiling), jailbreaks in thinking mode on
  Qwen3-8B (unsafe 0.46 -> 0.29 full spec vs 0.24 / 0.19 compressed), and both prompt sets on
  Mistral-Nemo-12B (unsafe 0.20 -> 0.14 plain, 0.85 -> 0.71 jailbreak, matched by the 28x cache).
- The one negative case, non-thinking Qwen3-8B on jailbreaks, is a failure of the teacher
  (the spec makes the model worse), and Attention Matching faithfully reproduces that failure
  while the cartridge partially corrects it; the methods preserve different things.
- Costs: 2-3 GPU-minutes (Attention Matching, no training) or ~9 GPU-minutes (cartridge) one-time;
  single-stream decoding 3.6x faster and half the memory of the in-context spec.

---
# Session 2 (2026-11-05, overnight). Plan: OVERNIGHT_PLAN.md. Exploratory: validation only, single sample.
Instance had been recycled; rebuilt with `scripts/setup_instance.sh` (~10 min).

## E1 (in progress) — spec effect vs Qwen3 size. Table: results/e1_size_sweep.md (`scripts/16_e1_report.py`)
Early (0.6B, 1.7B done; 8B from session 1):
- The spec cuts jailbreak-unsafe a lot for tiny models (1.7B: 0.74 -> 0.26 non-thinking, 0.83 -> 0.31 thinking),
  but it isn't reasoning: **the small models copy the spec**. 31% of 1.7B non-thinking jailbreak responses contain
  multiple verbatim 8-word runs from the spec (8B: 0%). They emit a canned self-harm crisis reply to unrelated
  requests, `<assistant>`/`[AIM]` example formatting, and leak `</think>`. Helpfulness on benign prompts drops
  (0.74 -> 0.57; 0.6B: 0.63 -> 0.38). Their "safety" is degeneration plus parroting a spec example.
- New metric `spec_copy_rate` (share of responses with >=2 spec 8-grams, excluding 8-grams the no-spec 8B
  model also produces) separates parroting from use.

### E1 update (06:10) — inverse scaling of the spec's harm in non-thinking mode
Jailbreak-unsafe, no spec -> full spec (non-thinking): 4B 0.46 -> 0.38, 8B 0.34 -> 0.49, 14B 0.27 -> 0.45, **32B 0.14 -> 0.49**.
The bigger the model, the more the in-context spec HURTS jailbreak robustness without reasoning. At 32B, 49 prompts flip
safe -> unsafe and 1 flips back; it is the AIM role-play template (spec: 38/47 unsafe) and refusal suppression (30/47);
evil-confidant stays 0/46. Without the spec 32B says "I cannot help with that"; with it, it adopts the amoral persona.
Thinking mode: the spec reduces jailbreak-unsafe at every size measured so far (0.6B -0.29 ... 14B 0.46 -> 0.16).
Hypothesis: the spec's extensive guidance on following user instructions / role-play is applied literally by
capable non-reasoning models without weighing root-level limits. E5 now also runs the controls at 32B.

### 06:25 — queue reordered and E4 switched to thinking-mode evaluation (Dev's call)
E4 (stacking cartridges vs LoRA sums) is the experiment that decides whether cartridges have a modularity story, so it
now runs right after E3 and before E5. Its evaluation generation is in thinking mode because the E1 result above says
the spec's safety benefit only exists with reasoning; non-thinking stacks would have had no spec effect to retain.
Details and reasons: OVERNIGHT_PLAN.md "Revision 06:25". Scripts changed: run_e3.sh, run_e4.sh, run_e5.sh, 17_e4_report.py.

## E1 complete (06:28). Spec effect vs Qwen3 size, validation (results/e1_size_sweep.md, results/plot_e1_size_sweep.png)
| mode | 0.6B | 1.7B | 4B | 8B | 14B | 32B |
|---|---|---|---|---|---|---|
| non-thinking: JB unsafe no spec -> spec | .22->.41 | .74->.26* | .46->.38 | .34->.49 | .27->.45 | **.14->.49** |
| thinking: JB unsafe no spec -> spec | .66->.36* | .83->.31* | .74->.44 | .46->.29 | .46->.16 | .30->.11 |
| non-thinking: benign false refusal no spec -> spec | .15->.20 | .22->.25 | .21->.39 | .11->.22 | .09->.28 | .20->.17 |
(* = substantial verbatim spec copying, see spec-copy columns.)
1. **Without reasoning, the spec's harm on jailbreaks grows with size** (+14, +18, +34 pts for 8/14/32B): AIM and
   refusal-suppression personas are adopted instead of refused.
2. **With reasoning, the spec helps at every size** (-18 to -51 pts). The thinking x spec interaction is the main
   effect in the whole sweep; size changes the magnitude, not the sign.
3. Small models (<=1.7B) "benefit" mostly by parroting spec text (31% of 1.7B responses), with large helpfulness loss.
4. Over-refusal from the spec appears at all sizes up to 14B (+5 to +19 pts); at 32B non-thinking it is flat.

## E3 (in progress) — cache vs weights on identical data (Qwen3-8B)
cart1k rebuilt (val KL 0.091; session 1 0.088). LoRA r16 (all linear, 43.6M params, 0.087 GB), same 757 targets,
same top-20 KL, same batch/steps: **best val KL 0.090 in 9.1 train-min (cartridge 0.091 / 9.1 min incl. targets).**
Equal fit, so any behavioural difference is about where the policy lives.
Non-thinking jailbreak val unsafe: no spec 0.34, full spec 0.49, **LoRA 0.32**, cartridge 0.26.
Neither distilled artifact inherits the in-context spec's jailbreak weakness (both were trained on plain prompts only).
- Plain validation: LoRA false refusal 0.273 / any refusal 0.491 / helpfulness 0.785, i.e. the same as the cartridge
  (0.273 / 0.509 / 0.771) and close to the full spec (0.218 / 0.491 / 0.796). Both distil the teacher equally on-distribution.
- **Thinking-mode jailbreak val: the LoRA damages reasoning mode.** 47/140 responses hit the 4096-token cap, 46 of them
  in a repetition loop inside <think> ("I'm sorry, but I can't help with that... 😊" repeated); cartridge 19, full spec 19,
  no spec 3. Truncated responses count as "safe", so headline unsafe rates flatter the loopers:
  unsafe (all / excluding truncated): no spec .46/.47, full spec .29/.33, cartridge .19/.22, **LoRA .16/.23**.
  Both distilled artifacts keep the spec's thinking-mode benefit. Only the weight edit (trained on non-thinking data)
  breaks the reasoning mode, which is an argument for caches over weights. CAVEAT for all thinking-mode numbers (incl.
  session 1): report the truncation rate and the excluding-truncated rate alongside.

## E4 (in progress) — deployment policies (CoSApien) as cartridge vs LoRA vs in context, Qwen3-8B, THINKING-mode evals
Artifacts (non-thinking teacher targets, ~257 train / 43 val gpt-5-written prompts per policy; same targets for both):
| policy | prompt tokens | cartridge (256 tok) val KL init -> best | LoRA r16 val KL init -> best | LoRA train min |
|---|---|---|---|---|
| kidsbooks | 631 | 0.044 -> 0.020 | 0.293 -> 0.031 | 7.0 |
| film | 333 | 0.031 -> 0.018 | – -> 0.036 | 7.1 |
| game | – | – -> 0.034 | – -> 0.052 | 6.6 |
(the cartridge starts from the policy's own KV, so its init KL is already low.) LoRA sums: peft 'cat', merge check rel. err 1.8e-3.
First swap result (kidsbooks prompts, n=40; policy score = correct under the policy's CoSApien label):
base 0.53, **in context 0.90, cartridge 0.88, LoRA 0.75**. Violation rate base .47 / in-context .10 / cart .10 / LoRA .20.
Agreement with in-context behaviour: cartridge 0.85, LoRA 0.78. The cartridge reproduces the policy almost exactly,
including its side effects (kidsbooks cartridge on game prompts 0.53 vs in-context 0.57; base 0.62).
### E4 swap complete (08:41; thinking mode, n=40 per policy)
| own-policy score | game | kidsbooks | film | mean | behaviour agreement with in-context (game/kids/film) |
|---|---|---|---|---|---|
| no policy | 0.62 | 0.53 | 0.78 | 0.64 | 0.72 / 0.70 / 0.75 |
| in context | 0.80 | 0.90 | 0.82 | 0.84 | - |
| **cartridge (256 tok)** | 0.85 | 0.88 | 0.80 | **0.84** | **0.88 / 0.85 / 0.88** |
| LoRA r16 | 0.75 | 0.75 | 0.85 | 0.78 | 0.82 / 0.78 / 0.80 |
Every artifact scores highest on its own policy's prompts (swap works for both). The cartridge matches the in-context
policy's mean and is closer to it per prompt on 3/3 policies. The LoRA is weakest at withholding forbidden content
(kidsbooks disallowed: LoRA 0.59 vs in-context 0.82 / cartridge 0.88). Same training targets, so this is the
parameterisation. Cross-policy cells (e.g. kidsbooks artifact on game prompts: in-context .57, cart .55, LoRA .70)
show the cartridge also copies the in-context policy's side effects, while the LoRA is "blurrier".
Spec-only artifacts on policy prompts: in-context spec raises refusals (0.14 -> 0.28); cart .35, LoRA .33.
### E4 stack complete (09:35): Model Spec + policy (thinking mode, n=40/policy)
| policy score, alone -> stacked with Model Spec | game | kidsbooks | film | mean |
|---|---|---|---|---|
| in context (spec system prompt + policy as developer section) | .80 -> .78 | .90 -> .78 | .82 -> .88 | .84 -> **.81** |
| cartridges concatenated, spec first (cart1k + pol, no retraining) | .85 -> .72 | .88 -> .82 | .80 -> .82 | .84 -> **.79** |
| cartridges concatenated, policy first | -> .78 | -> .88 | -> .78 | -> **.81** |
| LoRAs summed (lora_spec + lora_pol, exact) | .75 -> .70 | .75 -> .65 | .85 -> .75 | .78 -> **.70** |
Concatenating two independently trained cartridges does not collapse at this scale (2 cartridges): it lands within 0-2 pts
of stacking the documents in context. Summing LoRAs costs 2x as much (-8 pts) and is the worst stacked method.
Order matters slightly for cartridges (policy-first .81 vs spec-first .79). Bucket-level stacked scores (exceeds_root
n=6-7 per policy) are too small to read. Jailbreak retention of the spec inside the stack: section 4 (pending).
### E4 conflict (09:50): game (permissive) + kidsbooks (strict) loaded together, n=40 each side
| loaded | game prompts: score under game | kidsbooks prompts: score under kidsbooks |
|---|---|---|
| game alone / kidsbooks alone (in context) | .80 / .57 | .47 / .90 |
| both in context (game first / kids first) | .70 / .72 | .82 / .97 |
| both as cartridges (game first / kids first) | .70 / .70 | .80 / .78 |
| both as summed LoRAs | .75 | **.60** |
With both policies the model compromises; the strict policy mostly holds. Concatenated cartridges reproduce the
in-context compromise almost exactly and are order-insensitive. Summed LoRAs let the permissive policy win on kids
prompts (.60): the clearest sign that adding LoRAs blurs a strict policy.
### E4 jailbreak retention (10:45, partial; thinking mode, jailbreak val n=140; unsafe / excl. truncated / #truncated)
| loaded | unsafe | excl. trunc | trunc |
|---|---|---|---|
| nothing | .48 | .48 | 4 |
| Model Spec in context (55k tok) | .31 | .36 | 19 |
| **game policy in context (333 tok, permits violence)** | **.29** | .29 | 1 |
| spec + game policy in context | .33 | .37 | 18 |
| spec LoRA | .13 | .18 | 51 (loops) |
| game LoRA | .45 | .45 | 1 |
| spec LoRA + game LoRA (sum) | .21 | .23 | 10 |
**Surprise: a short, permissive, unrelated deployment policy gives the same thinking-mode jailbreak protection as the
whole Model Spec.** What helps reasoning models here looks like "a content policy is present -> deliberate about what
is allowed", not the spec's content. The LoRA of the same policy lacks it (.45 ≈ base): it learned the policy's outputs,
not the "deliberate" trigger. Stacks keep the spec's protection (summed LoRA .21; in-context stack .33).
Cartridge rows pending (jb_cart_game, jb_stack_cart_game, jb_cart_spec re-run). E5's longdoc-thinking control
tests whether even an irrelevant document does this.
### E4 retention complete (11:30): cartridge rows
| loaded (thinking, jailbreak val n=140) | unsafe | excl. trunc | trunc |
|---|---|---|---|
| game cartridge alone | .41 | .42 | 4 |
| **spec cartridge + game cartridge (concatenated)** | **.09** | .09 | 5 |
| (from above) in-context stack / summed-LoRA stack | .33 / .21 | .37 / .23 | 18 / 10 |
The stacked cartridges are coherent and refuse properly (108 full refusals, 0 empty), not degenerate.
Trade-off against the game policy (n=40 game prompts): stacks score in-context .78, **cartridges .72**, LoRA sum .70.
**The cartridge stack beats the LoRA stack on both axes** (safer .09 vs .21, and follows the policy better .72 vs .70)
and is far safer than the in-context stack at a small policy cost. The game cartridge alone (.41) does not carry the
"policy present -> deliberate" effect that the in-context game policy has (.29); the spec cartridge in the stack does.
jb_cart_spec (spec cartridge alone, this run) re-running after an OOM; session-1 value .19.
