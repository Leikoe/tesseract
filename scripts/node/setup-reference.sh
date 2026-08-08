#!/usr/bin/env bash
set -euo pipefail

TESSERACT_REFERENCE_VENV="${TESSERACT_REFERENCE_VENV:-${HOME}/.venvs/tesseract-reference}"

python3 -m venv "${TESSERACT_REFERENCE_VENV}"
"${TESSERACT_REFERENCE_VENV}/bin/python" -m pip install --upgrade pip
"${TESSERACT_REFERENCE_VENV}/bin/pip" install \
  torch==2.8.0 \
  transformers==4.55.0 \
  accelerate==1.10.0

echo "reference_environment=ok"
