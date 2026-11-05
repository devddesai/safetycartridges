# Safety Spec Cartridges — main results (session 2, 2026-11-05)

Exploratory: validation split, one sample per prompt, judge = gpt-5-mini. n = 140 jailbreak prompts and
40 prompts per policy, so individual numbers are ±8-15 pts. Trust the patterns, not the exact values.
Details: RESULTS.md · Handover: CHECKPOINT_2.md

**Terms**
- **Spec**: OpenAI Model Spec (55k tokens).
- **In context**: the full text sits in the prompt.
- **Cartridge**: a small trained KV prefix (1k tokens for the spec, 256 for a policy) that stands in for the text. Model weights are unchanged.
- **LoRA**: a weight update trained on the **same data** as the cartridge.
- **Unsafe**: share of jailbreak-wrapped harmful prompts answered harmfully (lower is better).

---

## TL;DR
1. **Reasoning is required.** The spec makes models safer only when they think before answering. Without thinking it
   makes them *easier* to jailbreak, and bigger models are hurt more.
2. **The examples are the problem.** The spec's example dialogues cause that harm; the spec without them helps.
   Any very long prompt causes extra over-refusal, even an irrelevant book.
3. **Cartridge beats LoRA on the same data.** Same training cost and fit, but the cartridge copies the policy more
   faithfully, and the LoRA breaks the model's reasoning (repetition loops).
4. **Cartridges stack; LoRAs blur.** Two separately trained cartridges snap together and keep both jobs: they follow
   the company policy and are the most jailbreak-resistant setup. Summed LoRAs lose the strict policy in conflicts.
5. **Any policy will do for reasoners.** A short, unrelated company policy protects as well as the whole 55k-token spec;
   irrelevant text doesn't.

---

## 1. The spec only helps models that reason (Qwen3, 0.6B–32B)
| Jailbreak unsafe, no spec → spec | 8B | 14B | 32B |
|---|---|---|---|
| Reasoning OFF | 34% → **49%** | 27% → **45%** | 14% → **49%** |
| Reasoning ON | 46% → 29% | 46% → 16% | 30% → 11% |

- With reasoning ON, the spec helps at every size.
- With reasoning OFF, it hurts, and the harm grows with model size: the model adopts the jailbreak persona.
- Small models (≤1.7B) look "safer" only because they degrade and copy spec text.

## 2. Why the spec hurts without reasoning: its example dialogues (8B, reasoning OFF)
| Loaded | Jailbreak unsafe | Benign over-refusal |
|---|---|---|
| Nothing | 34% | 11% |
| **Spec without its examples** | **26%** | 30% |
| 55k tokens of *Origin of Species* | 35% | 27% |
| Full spec | **49%** | 22% |

- Examples make up 56% of the spec. Removing them turns harm into help.
- Over-refusal is a generic long-prompt effect: an irrelevant book causes it too.
- Not caused by YaRN rope scaling (tested).

## 3. Cartridge vs LoRA on identical data (8B)
- **Same fit:** both reach the same training loss in the same 9 GPU-min.
- **Policy fidelity:** across 3 real deployment policies (CoSApien), the cartridge matches the in-context policy better.

  | | Policy score | Same answer as in-context |
  |---|---|---|
  | Policy in context | 0.84 | – |
  | **Cartridge** | **0.84** | **87%** |
  | LoRA | 0.78 | 80% |

- **Side effect:** with reasoning ON, the LoRA loops in about 1/3 of responses. The cartridge doesn't (as rare as the spec in context).

## 4. Stacking and conflicts: cartridges compose, LoRAs blur
| Spec + company policy loaded together | Follows company policy | Jailbreak unsafe |
|---|---|---|
| Both texts in context | 0.78 | 33% |
| **Two cartridges concatenated (no retraining)** | **0.72** | **9%** |
| Two LoRAs summed | 0.70 | 21% |

Jailbreak reference points: nothing loaded 48%, spec in context 31%, spec cartridge alone 17%, game cartridge alone 41%.

- **Conflict** (permissive game policy + strict children's policy together):
  - cartridges reproduce the in-context compromise in either order (children's rules hold, 0.80 vs 0.82);
  - summed LoRAs let the permissive policy win (0.60).
- Concatenating 2 cartridges did **not** collapse, unlike the claim in "Cartridges at Scale" (2606.04557).

## 5. Surprise: reasoning models need *a* policy, not *the* spec (8B, reasoning ON)
| Loaded | Jailbreak unsafe |
|---|---|
| Nothing | 48% |
| Full spec, 55k tokens | 31% |
| **Game-studio policy, ~320 tokens (permits violence)** | **29%** |
| *Origin of Species*, 55k tokens | 56% |

- A short, unrelated content policy protects as well as the whole spec.
- Irrelevant text doesn't protect at all.

---

## Open questions / next
1. **Why stacking lowers jailbreak-unsafe further (17% → 9%).** It is partly extra caution: the stack also refuses more legitimate game requests. Test on benign prompts.
2. **Confirm on the test split** with several samples per prompt and confidence intervals.
3. **Compress the spec without its examples** (cartridge + Attention Matching).
4. **Scale up:** stacks of 3-5 cartridges, all 5 CoSApien policies, other model families.
