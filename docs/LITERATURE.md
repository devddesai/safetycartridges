# Literature scan (2026-11-05, ~15 min automated web scan; VERIFY IDs before citing)

## Scooping check
- No paper found that compresses a safety spec into a cartridge / KV cache.
- Adjacent: AnchorKV (arXiv 2606.17872) and Safe-CAM (Ma et al., ACL 2026): KV eviction/compression changes safety behaviour.
- **Cartridges at Scale (arXiv 2606.04557, Amazon AGI): mixing independently trained cartridges causes collapse.** Directly tested by E4 (stacking).
- Closest policy-following competitors (all in-context or in-weights, none compressed): CoSA/CoSAlign (2410.08968), DynaGuard (2509.02563), gpt-oss-safeguard.

## By question
- Size/thinking x in-context spec: Deliberative Alignment (2412.16339); SpecBench/Align3 (2509.14760, test-time deliberation over 103 specs); RealGuardrails (2502.12197); HANDBOOK.md (2607.25398, long policy docs); AGENTIF (2505.16944); 2602.13562 (retrieve guidelines into context). No clean size-scaling study found, so E1 is a gap.
- Context distillation: Prompt Baking (2409.13697, KL baking, "half-baked" strength); Doc-to-LoRA (2602.15902); Askell et al. 2021 (2112.00861); Snell et al. 2022 (2209.15189).
- Composition: Cartridges (2506.06266, concat composition; prefix vs LoRA parameterisation §5.4); LoRA Soups (2410.13025); SafeMERGE (2503.17239); LoRA-for-safety in reasoning models (2507.17075); OSRM LoRA interference (ACL 2026 Findings). No study of safety under stacked LoRAs found.
- Prefix vs LoRA on the same data: mixed evidence (Novel-WD 2408.17070 favours prefix; Prefix-Tuning+ 2506.13674 and Program-as-Weights 2607.02512 favour LoRA).
- Reusable data: CoSApien, DynaBench, SpecBench, POLY-GUARD (2506.19054), RealGuardrails.
