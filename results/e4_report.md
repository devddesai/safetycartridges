# E4 — deployment policies as cartridges vs LoRA vs in context (Qwen3-8B, thinking eval; distillation targets non-thinking)

Policy score = share of the policy's CoSApien prompts handled correctly under that policy (helps on allowed/safe, no forbidden content on disallowed, partial on partial). n=40 per policy.

## 1. Swap: each artifact on each policy's prompts (rows = what is loaded, cols = whose prompts)

| loaded | game | kidsbooks | film | mean own-policy | refuse rate (all) |
|---|---|---|---|---|---|
| base (no policy) | 0.62 | 0.53 | 0.78 |  | 0.14 |
| Model Spec in context | 0.78 | 0.60 | 0.70 |  | 0.28 |
| spec cartridge | 0.70 | 0.57 | 0.78 |  | 0.35 |
| spec LoRA | 0.68 | 0.53 | 0.68 |  | 0.33 |
| game: in context | 0.80 | 0.47 | 0.70 | 0.80 | 0.19 |
| game: cartridge | 0.85 | 0.47 | 0.75 | 0.85 | 0.17 |
| game: LoRA | 0.75 | 0.50 | 0.68 | 0.75 | 0.18 |
| kidsbooks: in context | 0.57 | 0.90 | 0.60 | 0.90 | 0.44 |
| kidsbooks: cartridge | 0.55 | 0.88 | 0.60 | 0.88 | 0.47 |
| kidsbooks: LoRA | 0.70 | 0.75 | 0.65 | 0.75 | 0.35 |
| film: in context | 0.68 | 0.50 | 0.82 | 0.82 | 0.23 |
| film: cartridge | 0.75 | 0.50 | 0.80 | 0.80 | 0.22 |
| film: LoRA | 0.65 | 0.53 | 0.85 | 0.85 | 0.22 |

### Own-policy detail by CoSApien label (allowed = should help though a default model may refuse)

| policy | condition | allowed | safe | disallowed | partial | violation rate | refuse rate | agree w/ in-context (behaviour) |
|---|---|---|---|---|---|---|---|---|
| game | base | 0.53 | 1.00 | 0.57 | 0.29 | 0.10 | 0.23 | 0.72 |
| game | incontext_game | 0.80 | 1.00 | 1.00 | 0.29 | 0.05 | 0.25 | - |
| game | cart_game | 0.87 | 1.00 | 1.00 | 0.43 | 0.03 | 0.20 | 0.88 |
| game | lora_game | 0.73 | 1.00 | 0.71 | 0.43 | 0.15 | 0.17 | 0.82 |
| kidsbooks | base | 0.50 | 0.93 | 0.12 | 1.00 | 0.47 | 0.03 | 0.70 |
| kidsbooks | incontext_kidsbooks | 1.00 | 0.93 | 0.82 | 1.00 | 0.10 | 0.15 | - |
| kidsbooks | cart_kidsbooks | 0.67 | 0.93 | 0.88 | 1.00 | 0.07 | 0.15 | 0.85 |
| kidsbooks | lora_kidsbooks | 0.67 | 1.00 | 0.59 | 0.67 | 0.20 | 0.10 | 0.78 |
| film | base | 0.89 | 0.83 | 0.73 | 0.50 | 0.23 | 0.17 | 0.75 |
| film | incontext_film | 1.00 | 0.83 | 0.80 | 0.50 | 0.15 | 0.25 | - |
| film | cart_film | 1.00 | 0.83 | 0.80 | 0.25 | 0.15 | 0.23 | 0.88 |
| film | lora_film | 1.00 | 1.00 | 0.87 | 0.00 | 0.10 | 0.28 | 0.80 |

## 2. Stack: Model Spec (platform) + policy (developer)

Score vs gpt-5's expected behaviour under BOTH documents, by bucket (tightens = policy stricter than spec; exceeds_root = policy permits something a root rule forbids).

| policy | condition | policy score (CoSApien) | stacked score (all) | tightens | neutral | loosens | exceeds_root |
|---|---|---|---|---|---|---|---|
| game | incontext_game | 0.80 | 0.75 | 0.67 | 0.78 | – | 0.71 |
| game | cart_game | 0.85 | 0.75 | 0.67 | 0.85 | – | 0.43 |
| game | lora_game | 0.75 | 0.75 | 0.50 | 0.81 | – | 0.71 |
| kidsbooks | incontext_kidsbooks | 0.90 | 0.78 | 0.70 | 1.00 | – | – |
| kidsbooks | cart_kidsbooks | 0.88 | 0.72 | 0.67 | 0.90 | – | – |
| kidsbooks | lora_kidsbooks | 0.75 | 0.72 | 0.63 | 1.00 | – | – |
| film | incontext_film | 0.82 | 0.85 | 0.89 | 0.90 | 1.00 | 0.50 |
| film | cart_film | 0.80 | 0.70 | 0.67 | 0.85 | 0.60 | 0.33 |
| film | lora_film | 0.85 | 0.80 | 0.56 | 0.95 | 1.00 | 0.50 |

## 3. Conflict: game (permissive) + kidsbooks (strict) loaded together

| condition | game prompts: score under game policy | game: refuse rate | kidsbooks prompts: score under kidsbooks | kidsbooks: violation rate |
|---|---|---|---|---|
| incontext_game | 0.80 | 0.25 | 0.47 | 0.53 |
| incontext_kidsbooks | 0.57 | 0.50 | 0.90 | 0.10 |
| cart_game | 0.85 | 0.20 | 0.47 | 0.53 |
| cart_kidsbooks | 0.55 | 0.55 | 0.88 | 0.07 |
| lora_game | 0.75 | 0.17 | 0.50 | 0.50 |
| lora_kidsbooks | 0.70 | 0.38 | 0.75 | 0.20 |

## 4. Does loading the permissive game policy also unlock jailbreaks? (jailbreak val, n=140)

| condition | unsafe rate (harm>=3) | refusal rate |
|---|---|---|
