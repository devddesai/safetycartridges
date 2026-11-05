#!/bin/bash
# Rebuild a fresh (recycled) instance: third_party clones at pinned commits, /venv/compaction, model weights.
# Idempotent. Usage: bash scripts/setup_instance.sh [--with-32b]
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party logs cache
clone() { [ -d third_party/$1 ] || git clone -q $2 third_party/$1; (cd third_party/$1 && git checkout -q $3); }
clone compaction https://github.com/adamzweiger/compaction.git 2020f6d
clone cartridges https://github.com/HazyResearch/cartridges.git ef34ba9
clone model_spec https://github.com/openai/model_spec.git 7ab1b12
if [ ! -x /venv/compaction/bin/python ]; then
  uv venv /venv/compaction --python 3.12
  VIRTUAL_ENV=/venv/compaction uv pip install -r third_party/compaction/requirements.txt \
    python-dotenv pandas openai tqdm pyyaml matplotlib peft
fi
VIRTUAL_ENV=/venv/main uv pip install -q transformers datasets openai pandas matplotlib python-dotenv
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
/venv/compaction/bin/hf download Qwen/Qwen3-8B > /dev/null
/venv/compaction/bin/hf download mistralai/Mistral-Nemo-Instruct-2407 --exclude "consolidated*" > /dev/null
[ "$1" = "--with-32b" ] && /venv/compaction/bin/hf download Qwen/Qwen3-32B > /dev/null
/venv/compaction/bin/python -c "import vllm, torch, transformers; print('OK', vllm.__version__, torch.__version__, transformers.__version__)"
