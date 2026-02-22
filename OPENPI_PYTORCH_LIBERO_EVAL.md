# OpenPI PyTorch LIBERO Benchmark Evaluation — Agent Runbook

## Overview

This runbook sets up and runs the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO) using the **π₀.₅ (pi0.5) model** from [openpi](https://github.com/Physical-Intelligence/openpi) in **PyTorch mode** (non-Docker, non-JAX).

Architecture: Two processes communicate via WebSocket on port 8000:
- **Policy Server** — loads the PyTorch model, serves action predictions
- **Eval Client** — runs MuJoCo simulation, sends observations, receives actions

---

## System Requirements

- Ubuntu 22.04 / 24.04
- NVIDIA GPU ≥ 8 GB VRAM (tested: RTX 4090, CUDA 13.0)
- Python ≥ 3.11 (main env), Python 3.10 (LIBERO client env)
- `uv` package manager (pre-installed on vast.ai)
- `tmux` (pre-installed on vast.ai)
- ~20 GB disk (model checkpoint 6.8 GB + deps)

---

## Step 1: Clone Repository

```bash
# HTTPS clone (SSH may not be set up)
cd /workspace
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
```

> `GIT_LFS_SKIP_SMUDGE=1` avoids downloading large LFS files during submodule init.
> Submodules include `third_party/libero` and `third_party/aloha`.

---

## Step 2: Install Main Dependencies (Python 3.11 env)

```bash
cd /workspace/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install -e .
```

Installs ~200 packages including JAX, PyTorch 2.7.1, transformers 4.53.2.

---

## Step 3: Apply PyTorch Transformers Patches

Required for PyTorch model support (AdaRMS, precision control, KV cache fixes).

```bash
cd /workspace/openpi
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/
```

> WARNING: This permanently modifies the transformers library in the uv venv cache.
> To undo: `uv cache clean transformers`

---

## Step 4: Download JAX Checkpoint & Convert to PyTorch

```bash
cd /workspace/openpi

# Download JAX checkpoint (~5 GB, auto-cached to ~/.cache/openpi/)
uv run python -c "
from openpi.shared import download
ckpt = download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')
print('Downloaded to:', ckpt)
"

# Convert JAX → PyTorch (~6.8 GB safetensors output)
mkdir -p checkpoints
uv run examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_libero \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --output_path checkpoints/pi05_libero_pytorch
```

Expected output:
```
Model conversion completed successfully!
Model saved to checkpoints/pi05_libero_pytorch
```

Output files:
- `checkpoints/pi05_libero_pytorch/config.json`
- `checkpoints/pi05_libero_pytorch/model.safetensors` (6.8 GB)

---

## Step 5: Copy Norm Stats to PyTorch Checkpoint

The `serve_policy.py` script looks for norm stats in the checkpoint directory.
They are NOT automatically copied by the conversion script.

```bash
mkdir -p /workspace/openpi/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero
cp ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
   /workspace/openpi/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero/
```

> AGENT NOTE: This step is easy to miss. Without it, the server crashes with:
> `FileNotFoundError: Norm stats file not found at: .../norm_stats.json`

---

## Step 6: Setup LIBERO Client Virtual Environment (Python 3.10)

The eval client runs in a **separate venv** from the main openpi env.

```bash
cd /workspace/openpi

# Create Python 3.10 venv (uv downloads it automatically)
uv venv --python 3.10 examples/libero/.venv

# Bootstrap pip (uv venv does not include pip binary)
examples/libero/.venv/bin/python -m ensurepip --upgrade

PIP=examples/libero/.venv/bin/pip3

# Install PyTorch (use cu124 for CUDA 12.x/13.x — NOT the cu113 in requirements.txt)
$PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install LIBERO client dependencies
$PIP install \
    imageio imageio-ffmpeg \
    "numpy==1.22.4" \
    tqdm tyro PyYaml \
    "opencv-python==4.6.0.66" \
    "robosuite==1.4.1" \
    "matplotlib==3.5.3"

# Install third_party/libero requirements (skip egl_probe which fails on new CMake)
$PIP install \
    "hydra-core==1.2.0" \
    "wandb==0.13.1" \
    "easydict==1.9" \
    "einops==0.4.1" \
    "thop==0.1.1.post2209072238" \
    "robomimic==0.2.0" --no-deps \
    "transformers==4.21.1" --no-deps \
    "bddl==1.0.1" \
    "future==0.18.2" \
    "cloudpickle==2.1.0" \
    "gym==0.25.2" \
    "h5py" "tensorboard"

# Install libero package (from submodule, no deps)
$PIP install -e third_party/libero --no-deps

# Install openpi-client (WebSocket client library)
$PIP install -e packages/openpi-client
```

> AGENT NOTE: `egl_probe` fails to build on CMake ≥ 3.5+ (cmake_minimum_required issue).
> Skip it — it's used only for GPU EGL device probing, not needed for osmesa rendering.
>
> AGENT NOTE: `robomimic` and `transformers==4.21.1` must be installed with `--no-deps`
> to avoid pulling in `egl_probe` again.

---

## Step 7: Initialize LIBERO Config (Skip Interactive Prompt)

On first import, LIBERO asks interactively for dataset path. Pre-create the config:

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'EOF'
assets: /workspace/openpi/third_party/libero/libero/libero/assets
bddl_files: /workspace/openpi/third_party/libero/libero/libero/bddl_files
benchmark_root: /workspace/openpi/third_party/libero/libero/libero
datasets: /workspace/openpi/third_party/libero/libero/datasets
init_states: /workspace/openpi/third_party/libero/libero/libero/init_files
EOF
```

---

## Step 8: Install osmesa for Offscreen Rendering

EGL fails on this server (ZINK/Vulkan driver incompatible). Use osmesa instead.

```bash
apt-get install -y libosmesa6 libosmesa6-dev
```

> AGENT NOTE: Set `MUJOCO_GL=osmesa` when running the eval client (not `egl`).

---

## Step 9: Verify Client Environment

```bash
cd /workspace/openpi
PYTHONPATH=$PWD/third_party/libero \
MUJOCO_GL=osmesa \
examples/libero/.venv/bin/python -c "
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy
import torch, numpy
print('OK - torch:', torch.__version__, 'numpy:', numpy.__version__)
"
```

Expected: `OK - torch: 2.5.1+cu124 numpy: 1.22.4`

---

## Step 10: Run Evaluation (Two tmux Sessions)

### Policy Server (tmux session: `libero_server`)

```bash
tmux new-session -d -s libero_server -x 220 -y 50
tmux send-keys -t libero_server "cd /workspace/openpi && \
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /workspace/openpi/checkpoints/pi05_libero_pytorch \
    2>&1 | tee /workspace/server.log" Enter
```

Wait for: `INFO:websockets.server:server listening on 0.0.0.0:8000`

```bash
# Monitor server startup
tail -f /workspace/server.log
```

> NOTE: First inference triggers `torch.compile` — takes 3-5 minutes. Subsequent
> inferences are fast. The WebSocket client may timeout during compilation.

### Eval Clients (tmux sessions: `libero_eval_1`, `libero_eval_2`)

The 5 LIBERO task suites are split across 2 tmux sessions (run sequentially within each):

**Session `libero_eval_1`** — runs: libero_spatial, libero_object, libero_goal

```bash
EVAL_CMD='cd /workspace/openpi && PYTHONPATH=$PWD/third_party/libero MUJOCO_GL=osmesa \
/workspace/openpi/examples/libero/.venv/bin/python examples/libero/main.py \
--args.host 0.0.0.0 --args.port 8000 --args.num-trials-per-task 2 \
--args.task-suite-name'

tmux new-session -d -s libero_eval_1 -x 220 -y 50
tmux send-keys -t libero_eval_1 \
    "$EVAL_CMD libero_spatial 2>&1 | tee /workspace/eval_libero_spatial.log && \
     $EVAL_CMD libero_object  2>&1 | tee /workspace/eval_libero_object.log  && \
     $EVAL_CMD libero_goal    2>&1 | tee /workspace/eval_libero_goal.log    && \
     echo 'SESSION 1 DONE'" Enter
```

**Session `libero_eval_2`** — runs: libero_10, libero_90

```bash
tmux new-session -d -s libero_eval_2 -x 220 -y 50
tmux send-keys -t libero_eval_2 \
    "$EVAL_CMD libero_10 2>&1 | tee /workspace/eval_libero_10.log && \
     $EVAL_CMD libero_90 2>&1 | tee /workspace/eval_libero_90.log && \
     echo 'SESSION 2 DONE'" Enter
```

---

## Monitoring

```bash
# Server status
tail -f /workspace/server.log

# Eval progress (replace with target suite name)
tail -f /workspace/eval_libero_spatial.log

# Check all processes running
ps aux | grep -E "serve_policy|main\.py" | grep -v grep

# Attach to tmux session
tmux attach -t libero_server
tmux attach -t libero_eval_1
tmux attach -t libero_eval_2

# Detach from tmux: Ctrl+B then D
```

---

## Key File Paths (After Setup)

| Path | Description |
|------|-------------|
| `/workspace/openpi/` | Main repo |
| `/workspace/openpi/.venv/` | Python 3.11 main venv (server) |
| `/workspace/openpi/examples/libero/.venv/` | Python 3.10 LIBERO client venv |
| `/workspace/openpi/checkpoints/pi05_libero_pytorch/` | PyTorch checkpoint |
| `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/` | JAX checkpoint cache |
| `~/.libero/config.yaml` | LIBERO dataset path config |
| `/workspace/server.log` | Policy server log |
| `/workspace/eval_libero_*.log` | Per-suite eval logs |
| `/workspace/openpi/data/libero/videos/` | Rollout videos saved here |

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `FileNotFoundError: Norm stats file not found` | Run Step 5 (copy norm_stats.json) |
| `ModuleNotFoundError: No module named 'bddl'` | `pip install bddl==1.0.1 future` |
| `ModuleNotFoundError: No module named 'easydict'` | `pip install easydict==1.9` |
| `libEGL warning: egl: failed to create dri2 screen` | Use `MUJOCO_GL=osmesa` (not `egl`) |
| `egl_probe` CMake build error | Skip it (`--no-deps` on robomimic/transformers) |
| WebSocket keepalive timeout | Normal during first `torch.compile`, retry or wait |
| `pip: No such file or directory` (uv venv) | Use `python -m ensurepip` first, then use `pip3` |
| Interactive prompt on `from libero.libero import benchmark` | Pre-create `~/.libero/config.yaml` (Step 7) |
| `torch==1.11.0+cu113` not compatible | Replace with `torch` from `--index-url https://download.pytorch.org/whl/cu124` |

---

## Notes for AI Agents

- The `requirements.txt` in `examples/libero/` was compiled for Python 3.8 + CUDA 11.3.
  On modern hardware (CUDA 12+), **do not use it directly**. Follow Step 6 above instead.
- `uv pip install` and `pip install` inside a `uv venv` can conflict with the global env.
  Always use the explicit venv path: `examples/libero/.venv/bin/pip3`.
- The CLI arguments to `main.py` require `--args.` prefix (tyro wraps the `Args` dataclass
  under the function parameter name `args`).
- `torch.compile` on first inference takes 3-5 minutes on RTX 4090. Plan for this in timeouts.
