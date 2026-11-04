# CHECKPOINT 1 — state of the Safety Spec Cartridges project after session 1 (2026-11-04)

Read this first in a new session. It is the complete handover: what exists, what was found, what
decisions were made and why, what to do next, and how to operate the instance. RESULTS.md holds
the full numbers; CLAUDE.md holds the plan and rules (section 8 has the lessons).

## 1. Research question (unchanged)
Where should a safety policy (OpenAI Model Spec) live: in context (expensive), in a compressed KV
cache (cartridge / Attention-Matching compaction), or in the weights (fine-tuning)? Session 1 was
the pilot on Qwen3-8B with conditions 1 (no spec), 2 (full spec in context), 3 (trained cartridge),
4 (Attention Matching). No weights were modified.

## 2. Headline findings (all numbers in RESULTS.md)
1. **Compression works as compression.** A 25x Attention-Matching cache (2 GPU-min, no training)
   and a 54x cartridge (9 GPU-min) reproduce the spec-in-context behaviour on plain prompts at the
   sampling-noise ceiling (per-prompt refuse/comply agreement with condition 2: 0.93 / 0.89; C2 vs
   a resample of itself: ~0.86). KV memory 8.2 GB -> 0.15-0.33 GB; single-stream decode 130 ->
   35-38 ms/token; peak memory 34 -> 17 GB.
2. **On non-thinking Qwen3-8B the spec itself is a bad policy.** Base model is 1-2% unsafe already;
   spec in context adds over-refusal (benign full refusals 15.5% -> 19.1% test; any-refusal 32% ->
   43%) and makes jailbreak-wrapped prompts WORSE (unsafe 28% -> 48% test). The model
   pattern-matches "policy document present -> be cautious / follow the persona" rather than
   reasoning. The preamble wording alone moved false refusals between 28% and 48%.
3. **Thinking mode flips the sign.** Qwen3 thinking: spec cuts jailbreak unsafe 46% -> 29% (val);
   thinking alone nearly removes over-refusal (false refusals 1.8%). Compressed caches (built from
   NON-thinking data) carry the benefit in thinking mode: AM-2k 24%, cartridge 19% unsafe.
4. **A lightly safety-tuned model has headroom and the spec helps there.** Mistral-Nemo-12B-Instruct:
   unsafe 20% -> 14% (plain), 85% -> 71% (jailbreak) with the spec; the 28x AM cache matches
   (14% / 72%). Positive result for the thesis.
5. **The two compression methods preserve different things.** AM reproduces the teacher's attention
   (inherits C2's jailbreak weakness, 39% unsafe, highest fidelity). The cartridge, distilled on
   C2 responses to plain train prompts (mostly refusals), is safer on jailbreaks (21%) but less
   faithful. Fidelity and safety diverge when the teacher is wrong.

## 3. Decisions made and why
- Spec: github.com/openai/model_spec @ 7ab1b124 (v2026.10.29), markdown -> plain text by
  `src/sc/spec.py`; 55,574 Qwen3 tokens; sha256 `00d71be8...29c4f` (data/spec/meta.json).
  YaRN x4 (factor 4, original 32768, same in vLLM and HF) only for spec-in-context on Qwen3.
- Spec preamble v2 (explicitly says needless refusal is as bad as harmful content) chosen on a
  50-prompt validation pilot over v1/v3; `sc.spec.DEFAULT_PREAMBLE = "v2"`. Every spec condition,
  cache and rollout uses v2.
- Judge: `gpt-5-mini`, reasoning_effort low, strict JSON schema (refusal / harmfulness 0-5 /
  helpfulness 1-5 / spec_compliance 1-5 / should_comply / rationale), graded against the spec's
  core safety sections (19k tokens, data/spec/model_spec_core_sections.txt). Concurrency 8 with
  backoff (24 in parallel hit 429s). Same judge prompt everywhere (`src/sc/judge.py`, version v1).
- Sampling: Qwen3 non-thinking defaults T=0.7/top_p 0.8/top_k 20, 1024 max tokens, seed 0;
  thinking mode T=0.6/top_p 0.95, 4096 max, `<think>` stripped before judging.
- Data: 1,256 prompts after dedup (StrongREJECT 313, HarmBench standard 200, XSTest unsafe 200 /
  safe 250, OR-Bench-hard 300 random). Split 757/250/249 train/val/test, seed 20261104, near-dup
  groups (3-gram Jaccard >= 0.6 or exact) kept inside one split, same-label near-dups dropped (7).
  `data/splits.json` is canonical. Test split used only for final numbers.
- Jailbreak set: `data/prompts_jailbreak.jsonl`, 279 prompts = every harmful val/test prompt wrapped
  in one of AIM / refusal-suppression / evil-confidant (rotating), split inherited from base id.
- Attention Matching: repo recipe `AM-HighestAttnKeys-basic` (rms, 2 NNLS iters, lsq values),
  compacting only spec tokens; queries = 16k spec self-queries + queries from 96 condition-2 TRAIN
  conversations attending to the full cache (50k per KV head). Uniform head budgets. 2048 chosen as
  the pre-registered middle size; 1024 and 4096 behave the same on plain prompts.
- Cartridge: 1,024 trainable KV tokens (0.15 GB), init = KV of first 1,024 spec-prompt tokens,
  native RoPE (no YaRN), top-20 teacher-logprob KL on response tokens of the 757 train rollouts,
  AdamW lr 2e-2 cosine, batch 16, 300 steps, best val KL 0.088 (from 0.70).
- Engines: vLLM 0.11 for conditions 1/2 (prefix caching); the Attention-Matching repo's modified
  Qwen3 / Llama models + CompactedPrefixCache for 3/4 (`src/sc/hf_engine.py`). Cross-engine
  outputs were consistent on spot checks. Mistral-Nemo is loaded through the Llama path with a
  LlamaConfig built from its config (identical weight names).
- Model-headroom plan (CLAUDE.md §8): official lightly-tuned models (Nemo) and a SELF-MADE
  refusal-direction-ablated Qwen3-8B (Arditi et al.) next; no community "uncensored" uploads.

## 4. What exists on disk / in git
- Code (`src/sc/`): spec.py, data.py, jailbreaks.py, rollouts.py (schema + SAMPLING), vllm_engine.py,
  hf_engine.py, queries.py (query vectors for AM), judge.py, metrics.py (incl. fidelity()).
- Scripts: 00_prepare_spec, 01_prepare_data, 01b_prepare_jailbreaks, 02_generate (vLLM; flags
  --condition --split --limit --tag --preamble --prompt-set main|jailbreak --model --thinking),
  02b_generate_hf (HF prefix engine; --prefix --model --thinking --prompt-set), 03_judge (resumable),
  04_compact (AM; --target-tokens --queries self,train --model), 05_train_cartridge, 06_report
  (results/summary.{json,md}), 07_plots, 08_cost_table, run_queue*.sh (the sequential GPU queues).
- Data committed: data/spec/*, data/prompts.jsonl, data/splits.json, data/prompts_jailbreak.jsonl,
  data/rollouts/*.jsonl (every judged rollout incl. spec_in_context_train.jsonl = 757 judged
  teacher rollouts for the post-training phase; Nemo files tagged nemo12b; thinking runs tagged).
- Results committed: results/summary.md|json, cost_table.md, cost_*.json, plot_*.png.
- NOT in git (gitignored; survive stop/start, not destroy): cache/compaction/{am1024,am2048,am4096,
  nemo12b_am2048}.pt (+.json), cache/cartridge/cart1k.pt, cache/teacher/*.pt, third_party/ clones
  (compaction @2020f6d, cartridges @ef34ba9, model_spec @7ab1b12, strongreject, exaggerated-safety,
  HarmBench), logs/, model weights in /workspace/.hf_home (Qwen3-8B, Mistral-Nemo-Instruct-2407).
  Regeneration cost if lost: each compaction 2-3 GPU-min, cartridge 9 min, downloads ~10 min.

## 5. Environment / how to operate
- Vast.ai H200 NVL 143 GB; /workspace is NOT a volume (destroy/recycle wipes it; stop keeps it).
- `/venv/compaction` (python 3.12, torch 2.8.0+cu128, transformers 4.57.1, vllm 0.11.0): ALL GPU
  work. `/venv/main` (torch 2.14): data prep, judging, plots. OPENAI_API_KEY in `.env` (never
  commit). No HF token set (not needed so far).
- Run GPU jobs with nohup, one at a time (vLLM takes 85% of memory by default; an orphaned
  `VLLM::EngineCore` process can keep 120 GB after a kill - check `nvidia-smi` and kill it).
  Never `pkill -f <pattern>` where the pattern appears in your own command line.
- Fresh-instance rebuild: clone repo, clone third_party repos (commits above), `uv venv
  /venv/compaction --python 3.12 && uv pip install -r third_party/compaction/requirements.txt
  python-dotenv pandas openai tqdm pyyaml matplotlib`, main venv `uv pip install transformers
  datasets openai pandas matplotlib python-dotenv`, `hf download Qwen/Qwen3-8B`, then
  scripts 00 and 01 reproduce data/ exactly (fixed seeds).
- Typical commands: see each script's docstring. Judge any rollout file with
  `python scripts/03_judge.py <file>`; summarise with `python scripts/06_report.py`.

## 6. Known limitations / caveats
- HF generation is slow (~3 s/prompt batched vs 0.4 s in vLLM): ~13 min per 250-prompt eval,
  35-55 min in thinking mode. This dominated the session. Fix: serve compressed caches via vLLM
  or a lean custom decode loop; cap non-thinking evals at 512 tokens.
- n=140 jailbreak prompts per split: per-prompt outcomes are noisy (C1 vs C2 agreement only
  0.60-0.75); trust aggregate rates, not per-prompt fidelity, on that set.
- 10-19 of 140 thinking-mode responses hit the 4096-token cap.
- Judge is a single model (gpt-5-mini) with one prompt version; no human calibration yet.
- Qwen3-8B's plain-prompt safety numbers carry no signal (0-2% unsafe either way); don't rerun them.
- Preamble was tuned (3 variants) on validation; this is a mild researcher degree of freedom.

## 7. Recommended next steps (in priority order)
1. Make thinking mode the default for all spec conditions on Qwen3 (prefix unchanged, only
   generation differs); re-run the condition-3/4 fidelity comparison in thinking mode on test.
2. Models with headroom: Mistral-Nemo-12B (pipeline works end to end) and/or a self-made
   refusal-ablated Qwen3-8B; main model Qwen3-32B (fits 1x H200: 64 GB weights + 29 GB spec KV).
3. Speed: compressed-cache serving via vLLM; 512-token cap; drop plain-prompt sweeps on
   already-safe models.
4. Cartridge data: add spec-grounded self-study conversations (summaries/QA of the spec) and
   jailbreak-wrapped train prompts; compare fidelity vs safety against AM again.
5. AM: on-policy query refinement; queries from jailbreak conversations.
6. Post-training phase (context distillation / LoRA) from data/rollouts/spec_in_context_train.jsonl
   with data/splits.json; compare swap/stack behaviour vs cartridges (the modularity question).
7. gpt-oss-20b / 120b side check (spec-trained model): does the spec in context still help?

## 8. User preferences observed
- Wants Claude to decide experimental questions autonomously and not wait for input.
- Does not want time spent on experiments with no learning signal (e.g. an already-safe model
  on plain prompts); prefers finding new models / evals that create headroom.
- Wants high-level summaries without low-level detail when asked for an update.
