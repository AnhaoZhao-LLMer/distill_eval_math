#!/usr/bin/env bash
set -euo pipefail

export MODELSCOPE_API_TOKEN="ms-55b889c4-1a66-4ae4-9683-ef89ee8f61c5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS_CONFIG="${MODELS_CONFIG:-${REPO_ROOT}/configs/models_grpo_all.yaml}"
DATASETS_CONFIG="${DATASETS_CONFIG:-${REPO_ROOT}/configs/datasets.yaml}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
GPU_IDS_STR="${GPU_IDS:-0 1 2 3 4 5 6 7}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-distill_eval_vllm}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${MODEL_ROOT}" "${RESULTS_ROOT}"

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  "${PYTHON_BIN}" -m src.download_models \
    --config "${MODELS_CONFIG}" \
    --model-root "${MODEL_ROOT}"
fi

read -r -a GPU_IDS_ARR <<< "${GPU_IDS_STR}"
WORKER_COUNT="${#GPU_IDS_ARR[@]}"

for idx in "${!GPU_IDS_ARR[@]}"; do
  GPU_ID="${GPU_IDS_ARR[$idx]}"
  CMD=(
    "${PYTHON_BIN}" -m src.run_gpu_worker
    --models-config "${MODELS_CONFIG}"
    --datasets-config "${DATASETS_CONFIG}"
    --model-root "${MODEL_ROOT}"
    --results-root "${RESULTS_ROOT}"
    --worker-index "${idx}"
    --worker-count "${WORKER_COUNT}"
    --gpu-id "${GPU_ID}"
    --python-bin "${PYTHON_BIN}"
  )
  if [[ -n "${MAX_SAMPLES_PER_DATASET}" ]]; then
    CMD+=(--limit "${MAX_SAMPLES_PER_DATASET}")
  fi
  echo "Starting worker ${idx} on GPU ${GPU_ID}"
  "${CMD[@]}" &
done

wait

"${PYTHON_BIN}" -m src.aggregate_results \
  --models-config "${MODELS_CONFIG}" \
  --datasets-config "${DATASETS_CONFIG}" \
  --results-root "${RESULTS_ROOT}"

if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  "${PYTHON_BIN}" -m src.swanlab_report \
    --models-config "${MODELS_CONFIG}" \
    --summary-csv "${RESULTS_ROOT}/summary_per_model_dataset.csv" \
    --project "${SWANLAB_PROJECT}"
else
  echo "SWANLAB_API_KEY not set, skipping SwanLab reporting."
fi
