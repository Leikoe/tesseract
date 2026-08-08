#!/usr/bin/env bash
set -euo pipefail

TESSERACT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TESSERACT_REPO_PATH="${TESSERACT_REPO_PATH:-$(cd -- "${TESSERACT_SCRIPT_DIR}/../.." && pwd)}"
TESSERACT_RUST_TOOLCHAIN="${TESSERACT_RUST_TOOLCHAIN:-1.89.0}"
TESSERACT_CUDA_PATH="${TESSERACT_CUDA_PATH:-/usr/local/cuda-13.3}"
TESSERACT_MODEL_PATH="${TESSERACT_MODEL_PATH:-/home/ubuntu/models/Llama-3.2-1B-Instruct}"
TESSERACT_CUTILE_COMMIT="${TESSERACT_CUTILE_COMMIT:-9fe5756f861bc40f098e6981ac2dff6cf5d3d0e4}"
TESSERACT_CUTILE_PATH="${TESSERACT_CUTILE_PATH:-${HOME}/.cache/tesseract/cutile-rs}"
TESSERACT_CUBIN_CACHE_PATH="${XDG_CACHE_HOME:-${HOME}/.cache}/cutile/kernels"

# shellcheck source=/dev/null
. "${HOME}/.cargo/env"

test "$(rustc +"${TESSERACT_RUST_TOOLCHAIN}" --version | awk '{print $2}')" = "${TESSERACT_RUST_TOOLCHAIN}"
test -x "${TESSERACT_CUDA_PATH}/bin/nvcc"
command -v clang >/dev/null
ldconfig -p | grep -q 'libclang'
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
"${TESSERACT_CUDA_PATH}/bin/nvcc" --version | tail -n 1

if [[ -n "$(git -C "${TESSERACT_REPO_PATH}" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked files in ${TESSERACT_REPO_PATH} are dirty" >&2
  exit 1
fi

cd "${TESSERACT_REPO_PATH}"
cargo +"${TESSERACT_RUST_TOOLCHAIN}" test --all-targets
cargo +"${TESSERACT_RUST_TOOLCHAIN}" clippy --all-targets -- -D warnings
cargo +"${TESSERACT_RUST_TOOLCHAIN}" clippy --features cuda --all-targets -- -D warnings
cargo +"${TESSERACT_RUST_TOOLCHAIN}" run --release --bin model-check -- \
  --model-path "${TESSERACT_MODEL_PATH}"
cargo +"${TESSERACT_RUST_TOOLCHAIN}" run --release --features cuda \
  --bin cuda-check
cargo +"${TESSERACT_RUST_TOOLCHAIN}" run --release --features cuda \
  --bin model-cuda-check -- --model-path "${TESSERACT_MODEL_PATH}"
cargo +"${TESSERACT_RUST_TOOLCHAIN}" run --release --features cuda \
  --bin next-token-check -- --model-path "${TESSERACT_MODEL_PATH}" \
  --prompt "The capital of France is"

test -d "${TESSERACT_CUBIN_CACHE_PATH}"
test -n "$(find "${TESSERACT_CUBIN_CACHE_PATH}" -type f -name '*.cubin' -print -quit)"

mkdir -p "$(dirname -- "${TESSERACT_CUTILE_PATH}")"
if [[ ! -d "${TESSERACT_CUTILE_PATH}/.git" ]]; then
  git clone https://github.com/NVlabs/cutile-rs.git "${TESSERACT_CUTILE_PATH}"
fi
git -C "${TESSERACT_CUTILE_PATH}" fetch origin "${TESSERACT_CUTILE_COMMIT}"
git -C "${TESSERACT_CUTILE_PATH}" checkout --detach "${TESSERACT_CUTILE_COMMIT}"

cd "${TESSERACT_CUTILE_PATH}"
CUDA_TOOLKIT_PATH="${TESSERACT_CUDA_PATH}" \
  cargo +"${TESSERACT_RUST_TOOLCHAIN}" run --release \
  -p cutile-examples --example hello_world

echo "a100_node_validation=ok"
