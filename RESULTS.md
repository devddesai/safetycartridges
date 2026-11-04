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
