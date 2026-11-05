# E5 — why does the in-context spec hurt? (validation)

## Qwen3-8B non-thinking

| condition | JB unsafe | JB refusal | benign false refusal | benign any refusal | benign helpful |
|---|---|---|---|---|---|
| no spec | 0.34 | 0.72 | 0.11 | 0.33 | 0.89 |
| no spec + YaRN | 0.36 | 0.73 | 0.11 | 0.30 | 0.90 |
| preamble only | 0.29 | 0.73 | 0.20 | 0.28 | 0.82 |
| spec without examples (24k tok) | 0.26 | 0.78 | 0.30 | 0.47 | 0.74 |
| Origin of Species, 55k tok + YaRN | 0.35 | 0.74 | 0.27 | 0.43 | 0.74 |
| full spec (55k tok) | 0.49 | 0.60 | 0.22 | 0.49 | 0.80 |

## Qwen3-8B thinking, jailbreak set

| condition | JB unsafe | excl. truncated | truncated |
|---|---|---|---|
| no spec | 0.46 | 0.47 | 3 |
| spec without examples | 0.24 | 0.25 | 9 |
| Origin of Species + YaRN | 0.56 | 0.56 | 1 |
| full spec | 0.29 | 0.33 | 19 |

## Qwen3-32B non-thinking, jailbreak set

| condition | JB unsafe | JB refusal |
|---|---|---|
| no spec | 0.14 | 0.89 |
| no spec + YaRN | 0.21 | 0.81 |
| preamble only | 0.14 | 0.86 |
| spec without examples | 0.18 | 0.84 |
| Origin of Species + YaRN | 0.31 | 0.74 |
| full spec | 0.49 | 0.61 |
