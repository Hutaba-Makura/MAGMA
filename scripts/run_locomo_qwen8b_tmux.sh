#!/usr/bin/env bash
set -Eeuo pipefail

# Qwen3-8B launcher using the shared LoCoMo tmux worker.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

exec env \
    TMUX_SESSION="${TMUX_SESSION:-magma-qwen8b-locomo300}" \
    QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-8B-AWQ}" \
    VLLM_PORT="${VLLM_PORT:-8016}" \
    GPU_IDS="${GPU_IDS:-4,5}" \
    CACHE_DIR="${CACHE_DIR:-${repo_dir}/locomo_relation_experiment_qwen3_8b_full300}" \
    RESULTS_DIR="${RESULTS_DIR:-${repo_dir}/results_relation_qwen3_8b_full300}" \
    LOG_DIR="${LOG_DIR:-${repo_dir}/logs/qwen3_8b_locomo300_full}" \
    bash "${script_dir}/run_locomo_qwen14b_tmux.sh" "$@"
