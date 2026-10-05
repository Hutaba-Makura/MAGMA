#!/usr/bin/env bash
set -Eeuo pipefail

# Mistral-Small-24B launcher using the shared LoCoMo tmux worker.
# The cached checkpoint is BF16 (~47GB), so vLLM quantizes it to FP8 on load
# to fit two 16GB GPUs.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

exec env \
    TMUX_SESSION="${TMUX_SESSION:-magma-mistral24b-locomo300}" \
    QWEN_MODEL="${QWEN_MODEL:-mistralai/Mistral-Small-24B-Instruct-2501}" \
    VLLM_PORT="${VLLM_PORT:-8020}" \
    GPU_IDS="${GPU_IDS:-1,2}" \
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---quantization fp8}" \
    EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}" \
    CACHE_DIR="${CACHE_DIR:-${repo_dir}/locomo_relation_mistral24b_full300}" \
    RESULTS_DIR="${RESULTS_DIR:-${repo_dir}/results_relation_mistral24b_full300}" \
    LOG_DIR="${LOG_DIR:-${repo_dir}/logs/mistral24b_locomo300_full}" \
    bash "${script_dir}/run_locomo_qwen14b_tmux.sh" "$@"
