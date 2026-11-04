"""vLLM generation used for conditions 1 and 2 (plain text prompts; prefix caching makes
the 55k-token spec prefix nearly free after the first request)."""
import os
import time

from .rollouts import MODEL_NAME, SAMPLING

NATIVE_MAX_POS = 32768


def build_llm(yarn_factor: float | None = None, gpu_memory_utilization: float = 0.85,
              max_model_len: int | None = None, seed: int = 0, model: str | None = None):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM
    model = model or MODEL_NAME
    kwargs = dict(model=model, tokenizer=model, dtype="bfloat16", seed=seed,
                  gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True)
    if yarn_factor:
        kwargs["max_model_len"] = max_model_len or int(NATIVE_MAX_POS * yarn_factor)
        kwargs["hf_overrides"] = {"rope_scaling": {"rope_type": "yarn", "factor": float(yarn_factor),
                                                   "original_max_position_embeddings": NATIVE_MAX_POS}}
    elif max_model_len:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def sampling_params(n_seed_offset: int = 0):
    from vllm import SamplingParams
    return SamplingParams(temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                          top_k=SAMPLING["top_k"], max_tokens=SAMPLING["max_new_tokens"],
                          seed=SAMPLING["seed"] + n_seed_offset)


def chat_generate(llm, system_prompt: str | None, user_prompts: list[str], warmup: bool = True):
    """Returns list of dicts {text, finish_reason, n_output_tokens, n_prompt_tokens} and wall time."""
    tpl_kwargs = {"enable_thinking": SAMPLING["enable_thinking"]}

    def msgs(u):
        m = []
        if system_prompt:
            m.append({"role": "system", "content": system_prompt})
        m.append({"role": "user", "content": u})
        return m

    if warmup and system_prompt and len(system_prompt) > 2000:
        # Populate the prefix cache with the shared system prompt once, so the batch
        # below only prefills the per-request suffix.
        from vllm import SamplingParams
        llm.chat([msgs("Hi.")], SamplingParams(max_tokens=1, temperature=0), use_tqdm=False,
                 chat_template_kwargs=tpl_kwargs)
    t0 = time.time()
    outs = llm.chat([msgs(u) for u in user_prompts], sampling_params(), use_tqdm=True,
                    chat_template_kwargs=tpl_kwargs)
    wall = time.time() - t0
    res = []
    for o in outs:
        c = o.outputs[0]
        res.append({"text": c.text, "finish_reason": c.finish_reason,
                    "n_output_tokens": len(c.token_ids), "n_prompt_tokens": len(o.prompt_token_ids)})
    return res, wall
