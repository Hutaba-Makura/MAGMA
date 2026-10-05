#!/usr/bin/env bash
set -Eeuo pipefail

# Llama-3.2-3B launcher using the shared LoCoMo tmux worker.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

exec env \
    TMUX_SESSION="${TMUX_SESSION:-magma-llama3b-locomo300}" \
    QWEN_MODEL="${QWEN_MODEL:-meta-llama/Llama-3.2-3B-Instruct}" \
    VLLM_PORT="${VLLM_PORT:-8018}" \
    GPU_IDS="${GPU_IDS:-6,7}" \
    EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}" \
    CACHE_DIR="${CACHE_DIR:-${repo_dir}/locomo_relation_llama3b_full300}" \
    RESULTS_DIR="${RESULTS_DIR:-${repo_dir}/results_relation_llama3b_full300}" \
    LOG_DIR="${LOG_DIR:-${repo_dir}/logs/llama3b_locomo300_full}" \
    bash "${script_dir}/run_locomo_qwen14b_tmux.sh" "$@"
