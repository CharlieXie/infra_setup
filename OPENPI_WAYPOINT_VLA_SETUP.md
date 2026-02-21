# Openpi Waypoint VLA — 完整环境配置与训练启动指南

本文档面向 AI Agent，记录了在全新 vast.ai 服务器（双 GPU，推荐 2× RTX PRO 6000 Ada/Blackwell 96GB）上从零开始搭建 openpi Waypoint VLA 训练环境并启动训练的所有步骤。

> **设计原理**见 `/workspace/openpi/WAYPOINT_VLA_DESIGN.md`

---

## 目录

1. [机器要求](#1-机器要求)
2. [基础环境准备](#2-基础环境准备)
3. [克隆代码仓库](#3-克隆代码仓库)
4. [配置 Python 环境](#4-配置-python-环境)
5. [安装额外依赖](#5-安装额外依赖)
6. [验证 transformers 替换补丁](#6-验证-transformers-替换补丁)
7. [下载模型权重](#7-下载模型权重)
8. [下载和准备数据](#8-下载和准备数据)
9. [配置 wandb](#9-配置-wandb)
10. [启动 Action Expert 训练](#10-启动-action-expert-训练)
11. [启动 VLM 训练](#11-启动-vlm-训练)
12. [监控训练](#12-监控训练)
13. [常见问题与解决方案](#13-常见问题与解决方案)
14. [完整 Checklist](#14-完整-checklist)

---

## 1. 机器要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | 1× 80GB A100 | 2× RTX PRO 6000 Blackwell (97.9GB each) |
| RAM | 64 GB | 128 GB |
| 磁盘 | 200 GB | 500 GB SSD |
| CUDA | 12.0+ | 12.8 |
| OS | Ubuntu 22.04 | Ubuntu 22.04/24.04 |
| Python | 3.11 | 3.11 |

---

## 2. 基础环境准备

### 2.1 禁用 vast.ai 自动 tmux

```bash
touch ~/.no_auto_tmux
```

重连后生效。

### 2.2 配置 Git 身份

```bash
git config --global user.email "chuanliang.xie@gmail.com"
git config --global user.name "chuanliang"
```

### 2.3 安装 uv（openpi 使用 uv 管理依赖）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env   # 或者 source ~/.local/bin/env
uv --version          # 应该显示 uv 版本号
```

如果 `uv` 已在 `/usr/local/bin/uv`（vast.ai 预装），跳过此步。

---

## 3. 克隆代码仓库

```bash
cd /workspace

# 克隆 openpi（主要工作仓库，包含 waypoint VLA 实现）
git clone https://<PAT>@github.com/CharlieXie/openpi.git
# 如果是 fork/private 版本，替换仓库地址

# 克隆 infra_setup（本文档所在仓库）
git clone https://<PAT>@github.com/CharlieXie/infra_setup.git

# 替换 <PAT> 为实际的 GitHub Personal Access Token
```

验证：

```bash
ls /workspace/openpi/src/openpi/waypoint/   # 应该有 ae_model.py, vlm_model.py 等文件
ls /workspace/openpi/configs/               # 应该有 waypoint_ae_libero.yaml 等
```

---

## 4. 配置 Python 环境

openpi 使用 `uv` 管理虚拟环境。

```bash
cd /workspace/openpi

# 创建并安装 venv（基于 pyproject.toml，会自动安装所有依赖）
uv sync
```

这需要 10–20 分钟，会安装 JAX、PyTorch、transformers、Flax 等所有依赖。

验证：

```bash
.venv/bin/python --version              # 应该是 Python 3.11.x
.venv/bin/python -c "import torch; print(torch.__version__)"     # 2.7.x
.venv/bin/python -c "import jax; print(jax.__version__)"         # 0.5.x
.venv/bin/python -c "import transformers; print(transformers.__version__)"  # 4.53.x
```

---

## 5. 安装额外依赖

openpi 的默认 venv **不包含 TensorFlow**，但 RLDS 数据加载需要。必须手动安装：

```bash
cd /workspace/openpi

# 安装 TensorFlow 2.15.0（与 galaxea_0 一致，避免 tfds 版本冲突）
uv pip install --python .venv/bin/python \
    "tensorflow==2.15.0" \
    "tensorflow-datasets==4.9.3"
```

> **注意**: `tensorflow==2.20.0` 与当前 `ml_dtypes` 版本不兼容，会报 `AttributeError: module 'ml_dtypes' has no attribute 'int2'`。必须使用 2.15.0。

验证：

```bash
.venv/bin/python -c "
import tensorflow as tf
import tensorflow_datasets as tfds
print('TF:', tf.__version__, 'TFDS:', tfds.__version__)
"
# 期望输出: TF: 2.15.0 TFDS: 4.9.3
```

---

## 6. 验证 transformers 替换补丁

openpi 需要对 HuggingFace transformers 做 monkey-patch 来支持 AdaRMSNorm。检查补丁是否已应用：

```bash
.venv/bin/python -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly(), 'patch missing!'
print('transformers patch OK')
"
```

如果失败，应用补丁：

```bash
cp -r /workspace/openpi/src/openpi/models_pytorch/transformers_replace/* \
    /workspace/openpi/.venv/lib/python3.11/site-packages/transformers/
```

再次验证。

---

## 7. 下载模型权重

### 7.1 Pi0.5 base PyTorch 权重

权重路径: `/workspace/models/pi05_base_pytorch/`
需要的文件:
- `model.safetensors` (~14 GB, bfloat16, 3.6B 参数)
- `config.json`

**从 Google Drive 下载**（使用 rclone）:

```bash
# 首先配置 rclone（如果未配置）
rclone config   # 添加 Google Drive remote

# 下载权重
mkdir -p /workspace/models/pi05_base_pytorch
rclone copy gg1:models/pi05_base_pytorch/ /workspace/models/pi05_base_pytorch/ -P
```

**或者从 HuggingFace 下载原始 JAX 权重后转换**（如果 PyTorch 权重不可用）:

```bash
# 先下载 JAX checkpoint，然后用 openpi 的转换脚本
# 参考 openpi README 中的 "Convert to PyTorch" 章节
```

验证:

```bash
.venv/bin/python -c "
from safetensors.torch import load_file
t = load_file('/workspace/models/pi05_base_pytorch/model.safetensors', device='cpu')
print(f'Total keys: {len(t)}')   # 应该是 812
print('action_in_proj:', t['action_in_proj.weight'].shape)  # [1024, 32]
print('time_mlp_in:', t['time_mlp_in.weight'].shape)        # [1024, 1024]
"
```

---

## 8. 下载和准备数据

### 8.1 LIBERO 原始 RLDS 数据

原始数据来自 [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)。

```bash
mkdir -p /workspace/data/libero

# 使用 huggingface-cli 下载 libero_object（no noops 版本）
.venv/bin/python -c "
import tensorflow_datasets as tfds
import os
os.environ['HF_DATASETS_CACHE'] = '/workspace/data/.cache/huggingface'
# 将 RLDS 数据下载到指定目录
builder = tfds.builder('libero_object', data_dir='/workspace/data/libero/libero_object_no_noops')
builder.download_and_prepare()
"
```

**或者从已有机器 rsync**:

```bash
rsync -avz --progress \
    user@source:/workspace/data/libero/libero_object_no_noops/ \
    /workspace/data/libero/libero_object_no_noops/
```

目标目录结构（必须包含这几个文件）：

```
/workspace/data/libero/libero_object_no_noops/
└── libero_object_no_noops/
    └── 1.0.0/
        ├── dataset_info.json          ← TFDS 需要这个文件
        ├── features.json
        ├── dataset_statistics_*.json  ← 归一化统计量
        └── libero_object-train.tfrecord-*-of-00032   (32 个文件)
```

> **重要**: TFDS `builder_from_directory` 需要指向**直接含有 `dataset_info.json`** 的目录。
> 训练配置中写的是 `.../libero_object_no_noops/libero_object_no_noops/1.0.0`（两层嵌套）。

### 8.2 Waypoint-filtered RLDS 和 Waypoint Indices（VLM 训练 + AE 训练）

这两个文件由 G0 项目的 waypoint 提取脚本生成，可从已有机器复制：

```bash
# Waypoint 提取输出目录
rsync -avz --progress \
    user@source:/workspace/data/libero/libero_object_wp_001/ \
    /workspace/data/libero/libero_object_wp_001/
```

必须包含：

```
/workspace/data/libero/libero_object_wp_001/
├── waypoint_indices.json              ← AE 训练用：每个 episode 的 waypoint step 索引
└── waypoint_filtered_rlds__libero/
    └── 1.0.0/
        ├── dataset_info.json
        └── *.tfrecord                 ← VLM 训练用：每步都是一个 waypoint
```

验证数据可读：

```bash
cd /workspace/openpi
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from openpi.waypoint.normalize import load_dataset_statistics
stats = load_dataset_statistics('/workspace/data/libero_object_no_noops/1.0.0')
print('action q99 shape:', len(stats['action']['q99']))  # 7
print('proprio q99 shape:', len(stats['proprio']['q99']))  # 8
print('Data stats OK')
"
```

验证 RLDS 可读：

```bash
cd /workspace/openpi
.venv/bin/python -c "
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
b = tfds.builder_from_directory('/workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0')
ds = b.as_dataset(split='train')
ep = next(iter(ds))
steps = list(ep['steps'])
print(f'First episode: {len(steps)} steps, action shape: {steps[0][\"action\"].shape}')
# 期望: action shape: (7,)
"
```

---

## 9. 配置 wandb

```bash
cd /workspace/openpi
.venv/bin/python -m wandb login
# 输入 wandb API key（从 https://wandb.ai/settings 获取）
```

验证：

```bash
.venv/bin/python -c "import wandb; print(wandb.api.api_key[:8] + '...')"
```

---

## 10. 启动 Action Expert 训练

### 10.1 确认配置文件

检查 `/workspace/openpi/configs/waypoint_ae_libero.yaml` 中的路径：

```yaml
original_rlds_dir: /workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0
wp_indices_path: /workspace/data/libero/libero_object_wp_001/waypoint_indices.json
dataset_statistics_path: /workspace/data/libero_object_no_noops/1.0.0
pretrained_weight_path: /workspace/models/pi05_base_pytorch
```

确认这几个路径都存在：

```bash
ls /workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/dataset_info.json
ls /workspace/data/libero/libero_object_wp_001/waypoint_indices.json
ls /workspace/data/libero_object_no_noops/1.0.0/dataset_statistics*.json
ls /workspace/models/pi05_base_pytorch/model.safetensors
```

### 10.2 启动训练（tmux session）

```bash
cd /workspace/openpi
mkdir -p logs

# 新建 tmux session
tmux new-session -d -s waypoint_ae -x 220 -y 50

# 设置环境变量
tmux send-keys -t waypoint_ae "cd /workspace/openpi" Enter
tmux send-keys -t waypoint_ae "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter

# 启动双 GPU 训练
tmux send-keys -t waypoint_ae \
    ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_waypoint.py --mode ae \
    --config configs/waypoint_ae_libero.yaml \
    2>&1 | tee logs/waypoint_ae_libero.log" Enter
```

### 10.3 验证训练正常启动

```bash
# 等待约 90 秒后查看日志
sleep 90
tail -20 /workspace/openpi/logs/waypoint_ae_libero.log
```

正常启动的标志（按顺序出现）：
1. `WaypointAEDataset: 454 episodes, 8409 valid pairs` — 数据集加载成功
2. `Loaded 811 weight tensors, skipped 1` — Pi0.5 权重加载成功（time_mlp_in 因 shape 变化被跳过）
3. `Load dataset info from ... Constructing tf.data.Dataset` — RLDS 数据开始读取
4. `wandb: 🚀 View run at https://...` — wandb 连接成功
5. `Model: 3617.8M total, 3617.8M trainable` — 模型初始化成功
6. `[AE] step=0/20000 loss=0.7xxx lr=...` — **第一步 loss，训练开始！**

### 10.4 预期性能（2× RTX PRO 6000 Blackwell, batch_size=48）

| 指标 | 期望值 |
|------|--------|
| GPU 内存 (per GPU) | ~73 GB / ~63 GB |
| GPU 利用率 | ~100% |
| 速度 | ~7–9 s/step (全量 finetune 3.6B) |
| 初始 loss | ~0.7 |
| 总训练时间 | ~45–50 小时 (20000 steps) |

### 10.5 断点续训

```bash
tmux send-keys -t waypoint_ae \
    ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_waypoint.py --mode ae \
    --config configs/waypoint_ae_libero.yaml --resume \
    2>&1 | tee -a logs/waypoint_ae_libero.log" Enter
```

---

## 11. 启动 VLM 训练

VLM 训练使用 waypoint-filtered RLDS 数据，独立于 AE 训练。

### 11.1 确认 VLM 配置

```bash
cat /workspace/openpi/configs/waypoint_vlm_libero.yaml
# 关键路径:
#   wp_rlds_dir: /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero
#   dataset_statistics_path: /workspace/data/libero_object_no_noops/1.0.0
#   pretrained_weight_path: /workspace/models/pi05_base_pytorch
```

### 11.2 启动 VLM 训练

```bash
cd /workspace/openpi

tmux new-session -d -s waypoint_vlm -x 220 -y 50
tmux send-keys -t waypoint_vlm "cd /workspace/openpi" Enter
tmux send-keys -t waypoint_vlm "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter
tmux send-keys -t waypoint_vlm \
    ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_waypoint.py --mode vlm \
    --config configs/waypoint_vlm_libero.yaml \
    2>&1 | tee logs/waypoint_vlm_libero.log" Enter
```

---

## 12. 监控训练

### 实时日志

```bash
# AE 训练进度（只看关键行）
tail -f /workspace/openpi/logs/waypoint_ae_libero.log | grep "\[AE\]"

# VLM 训练进度
tail -f /workspace/openpi/logs/waypoint_vlm_libero.log | grep "\[VLM\]"
```

### GPU 状态

```bash
watch -n 5 nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

### wandb

训练启动后控制台会打印 wandb run URL，形如：
```
wandb: 🚀 View run at https://wandb.ai/<user>/waypoint_vla/runs/<run_id>
```

Project 名：`waypoint_vla`，包含 `train/loss`、`train/lr`、`train/grad_norm` 等指标。

### 检查 checkpoint 保存

默认每 2000 步保存一次 checkpoint：

```bash
ls /workspace/openpi/checkpoints/waypoint_ae_libero/
# 期望: 2000/ 4000/ ... 目录，各含 model.safetensors + optimizer.pt + metadata.pt
```

---

## 13. 常见问题与解决方案

### Q: `ModuleNotFoundError: No module named 'tensorflow'`

**原因**: openpi venv 默认不含 TensorFlow。

**解决**:
```bash
cd /workspace/openpi
uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"
```

### Q: `FileNotFoundError: Could not load dataset info from .../dataset_info.json`

**原因**: RLDS 路径指向了错误层级，应指向直接包含 `dataset_info.json` 的目录。

**解决**: 更新 `waypoint_ae_libero.yaml` 中的路径，确保末尾是 `1.0.0`：
```yaml
original_rlds_dir: /workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0
```

### Q: `RuntimeError: Error(s) in loading state_dict: size mismatch for time_mlp_in.weight`

**原因**: `safetensors.load_model(strict=False)` 不跳过 shape 不匹配的 key（只跳过 missing/unexpected）。

**解决**: 已在 `train_waypoint.py` 中用手动循环加载修复。确认代码中有以下逻辑：
```python
for name, param in state_dict.items():
    if own_state[name].shape != param.shape:
        logging.info(f"Skipping {name}: shape mismatch")
        continue
    own_state[name].copy_(param)
```

### Q: `RuntimeError: expected input[B, 224, 224, 3] to have 3 channels, but got 224`

**原因**: 图像张量格式为 BHWC 而 SigLIP 期望 BCHW。

**解决**: 确认 `ae_dataset.py` 和 `vlm_dataset.py` 的 collator 中有：
```python
imgs = imgs.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
```

### Q: `CUDA error: an illegal memory access` (DDP)

**原因**: NCCL 后端在该环境有兼容性问题。

**解决**: 确认 `train_waypoint.py` 中 DDP backend 为 `"gloo"`：
```python
backend = "gloo"
dist.init_process_group(backend=backend, init_method="env://")
```

### Q: `AttributeError: module 'ml_dtypes' has no attribute 'int2'`

**原因**: 安装了 TensorFlow 2.20.0 导致与 ml_dtypes 不兼容。

**解决**: 降级到 TF 2.15.0：
```bash
uv pip install --python .venv/bin/python "tensorflow==2.15.0"
```

### Q: 训练速度只有 0.1 steps/sec (GPU 利用率 0%)

**原因**: shuffle_buffer_size 过大，数据加载被阻塞（等待 buffer 满）。

**解决**: 减小 `shuffle_buffer_size` 到 200–500（LIBERO 数据集只有 ~8400 pairs per rank）：
```yaml
shuffle_buffer_size: 500
```

---

## 14. 完整 Checklist

```
□ touch ~/.no_auto_tmux
□ git config --global user.email / user.name
□ uv --version 可用 (或 /usr/local/bin/uv)
□ cd /workspace && git clone <openpi repo>
□ cd /workspace/openpi && uv sync  (10-20 分钟)
□ uv pip install "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"
□ transformers patch 验证通过
□ /workspace/models/pi05_base_pytorch/model.safetensors 存在 (14 GB)
□ /workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/dataset_info.json 存在
□ /workspace/data/libero/libero_object_wp_001/waypoint_indices.json 存在
□ /workspace/data/libero_object_no_noops/1.0.0/dataset_statistics_*.json 存在
□ .venv/bin/python -m wandb login (配置 API key)
□ 确认 configs/waypoint_ae_libero.yaml 所有路径正确
□ mkdir -p /workspace/openpi/logs
□ tmux new-session -d -s waypoint_ae 创建 session
□ 启动训练命令（见第 10.2 节）
□ 等待 90s，tail logs，确认 loss 出现
□ 检查 wandb dashboard
```

---

## 附录：关键路径速查

| 资源 | 路径 |
|------|------|
| openpi 代码 | `/workspace/openpi/` |
| Waypoint VLA 模块 | `/workspace/openpi/src/openpi/waypoint/` |
| 训练脚本 | `/workspace/openpi/scripts/train_waypoint.py` |
| AE 训练配置 | `/workspace/openpi/configs/waypoint_ae_libero.yaml` |
| VLM 训练配置 | `/workspace/openpi/configs/waypoint_vlm_libero.yaml` |
| Pi0.5 权重 | `/workspace/models/pi05_base_pytorch/` |
| LIBERO RLDS (原始) | `/workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/` |
| LIBERO 归一化统计 | `/workspace/data/libero_object_no_noops/1.0.0/dataset_statistics_*.json` |
| Waypoint indices | `/workspace/data/libero/libero_object_wp_001/waypoint_indices.json` |
| 训练日志 | `/workspace/openpi/logs/waypoint_ae_libero.log` |
| Checkpoints | `/workspace/openpi/checkpoints/waypoint_ae_libero/` |
| 设计文档 | `/workspace/openpi/WAYPOINT_VLA_DESIGN.md` |
