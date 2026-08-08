# A100 node setup

This is the reproducible setup for Tesseract's current spot-worker base image.
The node is disposable; the local Git repository and GitHub `main` are the
source of truth.

## Validated base image

- Ubuntu 22.04.5 LTS (Jammy), x86_64
- Linux 6.8
- NVIDIA A100-SXM4-80GB (`sm_80`)
- NVIDIA driver 580.126.09

The bootstrap installs the CUDA **toolkit only**. It deliberately does not
install a CUDA driver package or reboot the node.

## New-node procedure

From the local canonical checkout:

```bash
git push origin main
ssh ubuntu@NODE_IP 'git clone https://github.com/Leikoe/tesseract.git /home/ubuntu/tesseract'
scp .env ubuntu@NODE_IP:/home/ubuntu/tesseract/.env
ssh ubuntu@NODE_IP 'chmod 600 /home/ubuntu/tesseract/.env && /home/ubuntu/tesseract/scripts/node/bootstrap-a100.sh'
```

The `.env` is authorized for transfer to the A100 and contains `HF_TOKEN`. It
is ignored by Git and must remain mode `0600`.

Bootstrap is idempotent and installs:

- Rust 1.89.0 with rustup's minimal profile;
- CUDA Toolkit 13.3 without replacing the host driver;
- Clang and libclang for cuTile's bindgen step;
- Python venv support and an isolated Hugging Face CLI;
- `meta-llama/Llama-3.2-1B-Instruct` under `/home/ubuntu/models`.

Run all host/model/cuTile gates afterward:

```bash
ssh ubuntu@NODE_IP '/home/ubuntu/tesseract/scripts/node/verify-a100.sh'
```

## Normal development cycle

```text
local edit -> local tests -> commit -> push
                                  |
                                  v
                       node git pull --ff-only
                                  |
                                  v
                          node validation
                                  |
                                  v
                  scp unique evidence back local
```

Never author unique source code on the spot node. Profiles, benchmark CSVs, or
failure logs that matter must be copied into the local repository before they
are treated as durable evidence.

## Overrides

The scripts accept prefixed environment overrides:

| Variable | Default |
| --- | --- |
| `TESSERACT_REPO_PATH` | directory containing the script's checkout |
| `TESSERACT_RUST_TOOLCHAIN` | `1.89.0` |
| `TESSERACT_CUDA_PATH` | `/usr/local/cuda-13.3` |
| `TESSERACT_MODEL_ID` | `meta-llama/Llama-3.2-1B-Instruct` |
| `TESSERACT_MODEL_PATH` | `/home/ubuntu/models/Llama-3.2-1B-Instruct` |
| `TESSERACT_CUTILE_COMMIT` | known-good commit recorded in the scripts |
