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
