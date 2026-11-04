| condition | prefix tokens | prefix KV (GB) | batch-1 ms/token | batch-1 s / 128 tok | peak GB (batch 1) | one-time GPU-min |
|---|---|---|---|---|---|---|
| C1 no spec | 0 | 0.00 |  |  |  | 0 |
| C2 full spec (YaRN x4) | 55749 | 8.22 |  |  |  | 0 (prefill only) |
| C4 AM 4096 | 4271 | 0.63 | 34.5 | 4.42 | 17.8 | 3.0 |
| C4 AM 2048 | 2223 | 0.33 | 38.3 | 4.90 | 17.1 | 2.1 |
| C4 AM 1024 | 1199 | 0.18 | 37.6 | 4.82 | 16.8 | 1.7 |
| C3 cartridge 1024 | 1024 | 0.15 | 35.0 | 4.47 | 16.8 | 9.1 |
