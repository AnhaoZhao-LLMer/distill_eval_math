#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" && -x "/opt/miniconda3/bin/conda" ]]; then
  CONDA_BIN="/opt/miniconda3/bin/conda"
fi
CONDA_ENV_NAME="${CONDA_ENV_NAME:-distill_eval_vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CONDA_TEMPLATE_ENV="${CONDA_TEMPLATE_ENV:-}"
FORCE_REQUIREMENTS_SYNC="${FORCE_REQUIREMENTS_SYNC:-0}"

if [[ -z "${CONDA_BIN}" ]]; then
  echo "conda was not found in PATH. Set CONDA_BIN or activate a shell with conda available first." >&2
  exit 1
fi

CONDA_BASE="$("${CONDA_BIN}" info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

if [[ -z "${CONDA_TEMPLATE_ENV}" ]] && env_exists "kl_analysis"; then
  CONDA_TEMPLATE_ENV="kl_analysis"
fi

CREATED_FROM_TEMPLATE=0
if ! env_exists "${CONDA_ENV_NAME}"; then
  if [[ -n "${CONDA_TEMPLATE_ENV}" ]] && env_exists "${CONDA_TEMPLATE_ENV}"; then
    conda create -y -n "${CONDA_ENV_NAME}" --clone "${CONDA_TEMPLATE_ENV}"
    CREATED_FROM_TEMPLATE=1
  else
    conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
fi

if [[ "${CREATED_FROM_TEMPLATE}" == "1" && "${FORCE_REQUIREMENTS_SYNC}" != "1" ]]; then
  echo "Environment cloned from ${CONDA_TEMPLATE_ENV}; skipping pip sync by default."
  echo "Set FORCE_REQUIREMENTS_SYNC=1 if you want to re-run pip install -r requirements.txt."
else
  conda run -n "${CONDA_ENV_NAME}" python -m pip install --upgrade pip
  conda run -n "${CONDA_ENV_NAME}" python -m pip install -r "${REPO_ROOT}/requirements.txt"
fi

echo "Environment ready: ${CONDA_ENV_NAME}"
echo "Activate with: conda activate ${CONDA_ENV_NAME}"
