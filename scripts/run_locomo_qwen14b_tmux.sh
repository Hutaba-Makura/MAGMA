#!/usr/bin/env bash
set -Eeuo pipefail

# Launch a detached tmux job that serves Qwen3-14B-AWQ locally and evaluates
# the same reproducible 300-question stratified LoCoMo subset.

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session_name="${TMUX_SESSION:-magma-qwen14b-locomo50}"
model="${QWEN_MODEL:-Qwen/Qwen3-14B-AWQ}"
port="${VLLM_PORT:-8014}"
gpu_ids="${GPU_IDS:-2,3}"
max_model_len="${MAX_MODEL_LEN:-8192}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
client_python="${CLIENT_PYTHON:-/data/student/k2312068/.conda/envs/magma/bin/python}"
vllm_bin="${VLLM_BIN:-/data/student/k2312068/.conda/envs/gmemory-vllm/bin/vllm}"
embedding_device="${EMBEDDING_DEVICE:-cpu}"
cache_dir="${CACHE_DIR:-${repo_dir}/locomo_relation_experiment_qwen3_14b}"
results_dir="${RESULTS_DIR:-${repo_dir}/results_relation_qwen3_14b_full300}"
log_dir="${LOG_DIR:-${repo_dir}/logs/qwen3_14b_locomo50}"

# Cluster login environments commonly define HTTP(S) proxies. All vLLM
# traffic is local, so force both urllib/OpenAI and curl-compatible clients to
# bypass those proxies for the loopback endpoint.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

usage() {
    cat <<EOF
Usage:
  bash scripts/run_locomo_qwen14b_tmux.sh          Start detached run
  bash scripts/run_locomo_qwen14b_tmux.sh status   Show session and recent log
  bash scripts/run_locomo_qwen14b_tmux.sh attach   Attach to session

Optional environment variables:
  TMUX_SESSION, QWEN_MODEL, VLLM_PORT, GPU_IDS, MAX_MODEL_LEN,
  GPU_MEMORY_UTILIZATION, CLIENT_PYTHON, VLLM_BIN, EMBEDDING_DEVICE,
  CACHE_DIR, RESULTS_DIR
EOF
}

latest_run_log() {
    find "${log_dir}" -maxdepth 1 -type f -name 'run_*.log' -printf '%T@ %p\n' \
        2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

if [[ "${1:-}" == "status" ]]; then
    if tmux has-session -t "${session_name}" 2>/dev/null; then
        echo "Session ${session_name}: running"
    else
        echo "Session ${session_name}: not running"
    fi
    recent_log="$(latest_run_log || true)"
    if [[ -n "${recent_log}" ]]; then
        echo "Log: ${recent_log}"
        tail -n 40 "${recent_log}"
    fi
    exit 0
elif [[ "${1:-}" == "attach" ]]; then
    exec tmux attach-session -t "${session_name}"
elif [[ -n "${1:-}" && "${1:-}" != "--worker" ]]; then
    usage >&2
    exit 2
fi

if [[ "${1:-}" != "--worker" ]]; then
    command -v tmux >/dev/null || { echo "tmux is not installed" >&2; exit 1; }
    [[ -x "${client_python}" ]] || { echo "Missing client Python: ${client_python}" >&2; exit 1; }
    [[ -x "${vllm_bin}" ]] || { echo "Missing vLLM executable: ${vllm_bin}" >&2; exit 1; }
    if tmux has-session -t "${session_name}" 2>/dev/null; then
        echo "tmux session already exists: ${session_name}" >&2
        echo "Attach with: tmux attach -t ${session_name}" >&2
        exit 1
    fi

    mkdir -p "${log_dir}"
    run_log="${log_dir}/run_$(date +%Y%m%d_%H%M%S).log"
    tmux new-session -d -s "${session_name}" -c "${repo_dir}" \
        "exec env RUN_LOG='${run_log}' TMUX_SESSION='${session_name}' QWEN_MODEL='${model}' VLLM_PORT='${port}' GPU_IDS='${gpu_ids}' MAX_MODEL_LEN='${max_model_len}' GPU_MEMORY_UTILIZATION='${gpu_memory_utilization}' CLIENT_PYTHON='${client_python}' VLLM_BIN='${vllm_bin}' EMBEDDING_DEVICE='${embedding_device}' CACHE_DIR='${cache_dir}' RESULTS_DIR='${results_dir}' LOG_DIR='${log_dir}' bash '${script_path}' --worker"

    echo "Started detached tmux session: ${session_name}"
    echo "Model: ${model}"
    echo "GPUs: ${gpu_ids}"
    echo "Log: ${run_log}"
    echo "Attach: tmux attach -t ${session_name}"
    echo "Status: bash scripts/run_locomo_qwen14b_tmux.sh status"
    exit 0
fi

# The worker owns its log redirection. Keeping redirection out of tmux's shell
# command avoids quoting differences between interactive and detached shells.
if [[ -n "${RUN_LOG:-}" ]]; then
    exec >>"${RUN_LOG}" 2>&1
fi

cd "${repo_dir}"
mkdir -p "${log_dir}" "${cache_dir}" "${results_dir}"
server_log="${log_dir}/vllm_$(date +%Y%m%d_%H%M%S).log"
base_url="http://127.0.0.1:${port}/v1"
server_pid=""

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        echo "Stopping vLLM server (PID ${server_pid})"
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    if [[ ${status} -eq 0 ]]; then
        echo "Run completed successfully at $(date --iso-8601=seconds)"
    else
        echo "Run failed with status ${status} at $(date --iso-8601=seconds)" >&2
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
tensor_parallel_size="${#gpu_array[@]}"

echo "Started at: $(date --iso-8601=seconds)"
echo "Repository: ${repo_dir}"
echo "Model: ${model}"
echo "CUDA_VISIBLE_DEVICES: ${gpu_ids}"
echo "Tensor parallel size: ${tensor_parallel_size}"
echo "Endpoint: ${base_url}"
echo "Server log: ${server_log}"
echo "Results: ${results_dir}"

if ! "${client_python}" - "${port}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
then
    echo "Port ${port} is already in use; choose another VLLM_PORT." >&2
    exit 1
fi

echo "Starting vLLM (the first run also downloads the model)..."
CUDA_VISIBLE_DEVICES="${gpu_ids}" "${vllm_bin}" serve "${model}" \
    --served-model-name "${model}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --tensor-parallel-size "${tensor_parallel_size}" \
    --max-model-len "${max_model_len}" \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    --max-num-seqs 4 \
    --enable-prefix-caching \
    >"${server_log}" 2>&1 &
server_pid=$!

ready=0
for attempt in $(seq 1 540); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "vLLM exited before becoming ready. Last server log lines:" >&2
        tail -n 100 "${server_log}" >&2 || true
        exit 1
    fi
    if "${client_python}" - "${base_url}/models" "${model}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(sys.argv[1], timeout=3) as response:
    payload = json.load(response)
served = {item["id"] for item in payload.get("data", [])}
raise SystemExit(0 if sys.argv[2] in served else 1)
PY
    then
        ready=1
        break
    fi
    if (( attempt % 6 == 0 )); then
        echo "Waiting for vLLM: $((attempt * 10)) seconds elapsed"
    fi
    sleep 10
done

if [[ ${ready} -ne 1 ]]; then
    echo "vLLM did not become ready within 90 minutes." >&2
    tail -n 100 "${server_log}" >&2 || true
    exit 1
fi
echo "vLLM is ready."

common_args=(
    --dataset "${repo_dir}/data/locomo10.json"
    --sample 0 1 2 3 4 5 6 7 8 9
    --max-questions 30
    --balanced-categories
    --sampling-seed 20261001
    --resume
    --category-to-test 1,2,3,4,5
    --use-episodes
    --model "${model}"
    --llm-backend local
    --llm-base-url "${base_url}"
    --embedding-model minilm
    --cache-dir "${cache_dir}"
    --results-dir "${results_dir}"
    --n-workers 3
)

for mode in original free hybrid; do
    echo
    echo "===== Running relation mode: ${mode} ====="
    "${client_python}" "${repo_dir}/test_fixed_memory.py" \
        "${common_args[@]}" --relation-mode "${mode}"
done

echo
echo "===== Building three-mode comparison ====="
"${client_python}" "${repo_dir}/compare_relation_modes.py" \
    --results-dir "${results_dir}" \
    --sample 0 \
    --samples 0 1 2 3 4 5 6 7 8 9 \
    --output-prefix "${results_dir}/comparison_aggregate_samples_0_1_2_3_4_5_6_7_8_9"

echo "All three 300-question LoCoMo runs are complete."
