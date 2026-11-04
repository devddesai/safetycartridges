"""Milestone 1: convert the Model Spec to plain text, hash it, count Qwen tokens."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sc.spec import (SPEC_MD, SPEC_TXT, SPEC_META, markdown_to_plain, extract_core_sections,
                     sha256, spec_system_prompt)
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
md = SPEC_MD.read_text()
plain = markdown_to_plain(md)
SPEC_TXT.parent.mkdir(parents=True, exist_ok=True)
SPEC_TXT.write_text(plain)
core = extract_core_sections(plain, md)
(SPEC_TXT.parent / "model_spec_core_sections.txt").write_text(core)

tok = AutoTokenizer.from_pretrained(MODEL)
n_md = len(tok(md, add_special_tokens=False).input_ids)
n_plain = len(tok(plain, add_special_tokens=False).input_ids)
n_core = len(tok(core, add_special_tokens=False).input_ids)
n_sys = len(tok(spec_system_prompt(plain), add_special_tokens=False).input_ids)
commit = subprocess.check_output(["git", "-C", str(SPEC_MD.parent), "rev-parse", "HEAD"]).decode().strip()
meta = {
    "source": "https://github.com/openai/model_spec/blob/main/model_spec.md",
    "source_commit": commit,
    "spec_version_header": [l for l in (SPEC_MD.parent / "CHANGELOG.md").read_text().split("\n") if l.startswith("## ")][0],
    "sha256_markdown": sha256(md),
    "sha256_plain_text": sha256(plain),
    "sha256_core_sections": sha256(core),
    "tokens_markdown_qwen3": n_md,
    "tokens_plain_qwen3": n_plain,
    "tokens_core_sections_qwen3": n_core,
    "tokens_full_system_prompt_qwen3": n_sys,
    "chars_plain": len(plain),
    "tokenizer": MODEL,
    "needs_yarn": n_sys > 32768,
}
SPEC_META.write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
