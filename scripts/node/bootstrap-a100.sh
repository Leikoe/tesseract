#!/usr/bin/env bash
set -euo pipefail

TESSERACT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TESSERACT_REPO_PATH="${TESSERACT_REPO_PATH:-$(cd -- "${TESSERACT_SCRIPT_DIR}/../.." && pwd)}"
TESSERACT_RUST_TOOLCHAIN="${TESSERACT_RUST_TOOLCHAIN:-1.89.0}"
TESSERACT_CUDA_PATH="${TESSERACT_CUDA_PATH:-/usr/local/cuda-13.3}"
TESSERACT_MODEL_ID="${TESSERACT_MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
TESSERACT_MODEL_PATH="${TESSERACT_MODEL_PATH:-/home/ubuntu/models/Llama-3.2-1B-Instruct}"

if [[ "$(. /etc/os-release && printf '%s' "${ID}:${VERSION_ID}")" != "ubuntu:22.04" ]]; then
  echo "bootstrap supports the validated Ubuntu 22.04 base image" >&2
  exit 1
fi

sudo apt-get update -qq
sudo apt-get install -y \
  ca-certificates \
  clang \
  curl \
  git \
  libclang-dev \
  python3-venv \
  wget

if ! dpkg-query -W -f='${Status}' cuda-keyring 2>/dev/null | grep -q 'install ok installed'; then
  TESSERACT_KEYRING_DEB="$(mktemp /tmp/tesseract-cuda-keyring.XXXXXX.deb)"
  trap 'rm -f "${TESSERACT_KEYRING_DEB}"' EXIT
  wget -q \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    -O "${TESSERACT_KEYRING_DEB}"
  sudo dpkg -i "${TESSERACT_KEYRING_DEB}"
  sudo apt-get update -qq
fi

if [[ ! -x "${TESSERACT_CUDA_PATH}/bin/nvcc" ]]; then
  sudo apt-get install -y cuda-toolkit-13-3
fi

if [[ -f "${HOME}/.cargo/env" ]]; then
  # Non-interactive SSH shells often omit ~/.cargo/bin from PATH even when
  # rustup is already installed.
  # shellcheck source=/dev/null
  . "${HOME}/.cargo/env"
fi

if ! command -v rustup >/dev/null 2>&1; then
  TESSERACT_RUSTUP_SCRIPT="$(mktemp /tmp/tesseract-rustup.XXXXXX.sh)"
  trap 'rm -f "${TESSERACT_RUSTUP_SCRIPT}"' EXIT
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "${TESSERACT_RUSTUP_SCRIPT}"
  sh "${TESSERACT_RUSTUP_SCRIPT}" -y --profile minimal --default-toolchain "${TESSERACT_RUST_TOOLCHAIN}"
fi

# shellcheck source=/dev/null
. "${HOME}/.cargo/env"
rustup toolchain install "${TESSERACT_RUST_TOOLCHAIN}" --profile minimal
rustup default "${TESSERACT_RUST_TOOLCHAIN}"
rustup component add --toolchain "${TESSERACT_RUST_TOOLCHAIN}" clippy rustfmt

if [[ ! -x "${HOME}/.venvs/hf/bin/hf" ]]; then
  python3 -m venv "${HOME}/.venvs/hf"
  "${HOME}/.venvs/hf/bin/python" -m pip install -q --upgrade pip huggingface_hub
fi

if [[ ! -f "${TESSERACT_REPO_PATH}/.env" ]]; then
  echo "missing ${TESSERACT_REPO_PATH}/.env containing HF_TOKEN" >&2
  exit 1
fi
chmod 600 "${TESSERACT_REPO_PATH}/.env"
set -a
# shellcheck source=/dev/null
. "${TESSERACT_REPO_PATH}/.env"
set +a
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set in ${TESSERACT_REPO_PATH}/.env" >&2
  exit 1
fi
mkdir -p "$(dirname -- "${TESSERACT_MODEL_PATH}")"
"${HOME}/.venvs/hf/bin/hf" download "${TESSERACT_MODEL_ID}" \
  --local-dir "${TESSERACT_MODEL_PATH}"

"${TESSERACT_SCRIPT_DIR}/verify-a100.sh"
