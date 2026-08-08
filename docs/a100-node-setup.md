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

Verification compiles and executes both Tesseract's own BF16 cuTile kernel and
the pinned upstream cuTile hello-world kernel. This distinguishes an upstream
toolchain installation check from a project-owned BF16 execution check.

Tesseract explicitly enables cuTile's persistent runtime CUBIN cache before
launching kernels. The cache defaults to
`$XDG_CACHE_HOME/cutile/kernels` or `~/.cache/cutile/kernels`, is private to the
node user, and has a 2 GiB soft capacity. The verifier requires at least one
cached CUBIN after the project validation runs. Preserve or mount this directory
on future nodes when cold-start latency matters; its content is derived and
must never replace Git as the source of truth.

The separate cuda-tile/LLVM build cache is enabled in `.cargo/config.toml` so
`cargo clean` does not force the expensive toolchain build to start over.

Run all host/model/cuTile gates afterward:

```bash
ssh ubuntu@NODE_IP '/home/ubuntu/tesseract/scripts/node/verify-a100.sh'
```

For the slower memory-safety gate, build the CUDA smoke and next-token tools and
run them under the CUDA toolkit's Compute Sanitizer:

```bash
cargo build --release --features cuda --bin cuda-check --bin next-token-check
compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./target/release/cuda-check
compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./target/release/next-token-check \
  --model-path /home/ubuntu/models/Llama-3.2-1B-Instruct \
  --prompt "The capital of France is"
```

For the independent pinned PyTorch/Transformers correctness gate, install the
isolated reference environment once, then compare three fixed prompts:

```bash
scripts/node/setup-reference.sh
~/.venvs/tesseract-reference/bin/python scripts/reference/llama_logits.py \
  --model-path /home/ubuntu/models/Llama-3.2-1B-Instruct \
  --tesseract-bin target/release/next-token-check
```

The reference packages are deliberately isolated from the production server;
they are not runtime dependencies.

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
| `TESSERACT_MODEL_REVISION` | `9213176726f574b556790deb65791e0c5aa438b6` |
| `TESSERACT_MODEL_PATH` | `/home/ubuntu/models/Llama-3.2-1B-Instruct` |
| `TESSERACT_CUTILE_COMMIT` | known-good commit recorded in the scripts |
