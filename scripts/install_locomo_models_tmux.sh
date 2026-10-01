#!/usr/bin/env bash
set -u

session_name="${TMUX_SESSION:-magma-locomo-model-install}"
python_bin="${HF_PYTHON:-/data/student/k2312068/.conda/envs/gmemory-vllm/bin/python}"
log_dir="${LOG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs/model_install}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "status" ]]; then
    tmux has-session -t "${session_name}" 2>/dev/null && echo "Session ${session_name}: running" || echo "Session ${session_name}: not running"
    recent_log="$(find "${log_dir}" -maxdepth 1 -type f -name 'install_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -n "${recent_log}" ]]; then
        echo "Log: ${recent_log}"
        tail -n 40 "${recent_log}"
    fi
    exit 0
fi

if [[ "${1:-}" == "attach" ]]; then
    exec tmux attach-session -t "${session_name}"
fi

if [[ "${1:-}" != "--worker" ]]; then
    command -v tmux >/dev/null || { echo "tmux is not installed" >&2; exit 1; }
    [[ -x "${python_bin}" ]] || { echo "Missing Python: ${python_bin}" >&2; exit 1; }
    if tmux has-session -t "${session_name}" 2>/dev/null; then
        echo "tmux session already exists: ${session_name}" >&2
        exit 1
    fi

    mkdir -p "${log_dir}"
    run_log="${log_dir}/install_$(date +%Y%m%d_%H%M%S).log"
    tmux new-session -d -s "${session_name}" -c "${repo_dir}" \
        "exec env RUN_LOG='${run_log}' HF_PYTHON='${python_bin}' bash '${BASH_SOURCE[0]}' --worker"

    echo "Started detached tmux session: ${session_name}"
    echo "Log: ${run_log}"
    echo "Status: bash scripts/install_locomo_models_tmux.sh status"
    exit 0
fi

if [[ -n "${RUN_LOG:-}" ]]; then
    exec >>"${RUN_LOG}" 2>&1
fi

models=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "openai/gpt-oss-20b"
    "mistralai/Mistral-Small-24B-Instruct-2501"
)

for model in "${models[@]}"; do
    echo "===== Checking cache for ${model} ====="
    if "${HF_PYTHON}" - "${model}" <<'PY'
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

model_id = sys.argv[1]
try:
    snapshot_download(repo_id=model_id, local_files_only=True)
except (LocalEntryNotFoundError, OSError):
    raise SystemExit(1)
PY
    then
        echo "Already cached: ${model}"
        continue
    fi

    echo "===== Downloading or resuming ${model} ====="
    "${HF_PYTHON}" - "${model}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id = sys.argv[1]
snapshot_download(repo_id=model_id)
print(f"Downloaded: {model_id}")
PY
    status=$?
    if [[ ${status} -ne 0 ]]; then
        echo "FAILED (${status}): ${model}"
    fi
done

echo "Model installation session finished at $(date --iso-8601=seconds)"
