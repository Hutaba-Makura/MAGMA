#!/usr/bin/env bash
set -Eeuo pipefail

# gpt-oss-20b launcher using the shared LoCoMo tmux worker.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

exec env \
    TMUX_SESSION="${TMUX_SESSION:-magma-gptoss20b-locomo300}" \
    QWEN_MODEL="${QWEN_MODEL:-openai/gpt-oss-20b}" \
    VLLM_PORT="${VLLM_PORT:-8020}" \
    GPU_IDS="${GPU_IDS:-1,2}" \
    EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}" \
    CACHE_DIR="${CACHE_DIR:-${repo_dir}/locomo_relation_gptoss20b_full300}" \
    RESULTS_DIR="${RESULTS_DIR:-${repo_dir}/results_relation_gptoss20b_full300}" \
    LOG_DIR="${LOG_DIR:-${repo_dir}/logs/gptoss20b_locomo300_full}" \
    bash "${script_dir}/run_locomo_qwen14b_tmux.sh" "$@"
