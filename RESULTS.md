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
