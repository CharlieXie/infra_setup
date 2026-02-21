# Galaxea-0 Libero 项目完整配置文档

本文档记录了在新服务器（vast.ai / 无桌面 Linux）上从零开始搭建 `galaxea_0` 项目、用 Libero 数据 finetune 模型、并运行评估的所有步骤和已知问题。AI 可以按本文档的顺序帮用户完整复现配置。

---

## 目录

1. [项目结构说明](#1-项目结构说明)
2. [第一步：克隆仓库](#2-第一步克隆仓库)
3. [第二步：创建 conda 环境](#3-第二步创建-conda-环境)
4. [第三步：下载模型权重（Google Drive）](#4-第三步下载模型权重google-drive)
5. [第四步：下载训练数据集（HuggingFace）](#5-第四步下载训练数据集huggingface)
6. [第五步：修改训练配置并启动 Finetune](#6-第五步修改训练配置并启动-finetune)
7. [第六步：运行 Libero 评估](#7-第六步运行-libero-评估)
8. [评估环境已知问题与解决方案](#8-评估环境已知问题与解决方案)

---

## 1. 项目结构说明

```
/workspace/
├── galaxea_0/                  # 主项目仓库（本文档的核心）
│   ├── env.yaml                # conda 环境定义文件
│   ├── finetune.py             # finetune 入口脚本
│   ├── google/                 # PaliGemma 基础模型（从 Google Drive 下载后拷贝至此）
│   │   └── paligemma-3b-pt-224/
│   ├── runs/                   # 训练输出目录（checkpoint 保存在此）
│   ├── experiments/robot/libero/
│   │   ├── run_libero_eval.py  # 评估入口脚本
│   │   └── run_libero_eval_single.py
│   └── vla/config/allen/libero_ft/
│       └── ft_libero_pi_object.yml  # 主配置文件（训练 + 评估）
├── models/
│   ├── google/                 # PaliGemma 基础模型（从 Google Drive 下载）
│   │   └── paligemma-3b-pt-224/
│   └── runs_og/                # 预训练 checkpoint（从 Google Drive 下载）
│       └── pick_and_place_ckpt/
│           └── model_100000.pt
├── data/
│   └── libero_object_no_noops/ # Libero Object 数据集（从 HuggingFace 下载）
│       └── 1.0.0/
│           ├── dataset_info.json
│           ├── features.json
│           └── libero_object-train.tfrecord-XXXXX-of-00032  # 32 个文件
└── openpi/
    └── third_party/libero/     # libero 仿真环境源码（评估用）
```

---

## 2. 第一步：克隆仓库

```bash
cd /workspace

# 使用 PAT token 克隆（替换 <PAT> 为实际 token）
git clone https://<PAT>@github.com/CharlieXie/galaxea_0.git

# 切换到 main 分支
cd galaxea_0
git checkout main
# 如果 main 不存在本地，用：git checkout -b main origin/main
```

---

## 3. 第二步：创建 conda 环境

> **前提**：服务器已安装 miniforge/anaconda，Python 3.10，CUDA 12.8。

### 3.1 用 env.yaml 创建环境

```bash
cd /workspace/galaxea_0
conda env create -f env.yaml
# 耗时约 5-10 分钟，环境安装在 /venv/py10_g0_train
```

> `env.yaml` 中有两个被注释掉的包，需要手动安装（见下）。

### 3.2 手动安装 dlimp

```bash
cd /workspace
git clone https://github.com/kvablack/dlimp.git

source /opt/miniforge3/etc/profile.d/conda.sh && conda activate py10_g0_train
pip install -e /workspace/dlimp
```

### 3.3 手动安装 openpi-client

`openpi-client` 在仓库内的 `openpi-client/` 目录中：

```bash
conda activate py10_g0_train
pip install -e /workspace/galaxea_0/openpi-client
```

### 3.4 验证环境

```bash
conda activate py10_g0_train
python -c "import torch; print('torch:', torch.__version__)"
# 期望输出：torch: 2.10.0+cu128
python -c "import dlimp; import openpi_client; print('ok')"
```

---

## 4. 第三步：下载模型权重（Google Drive）

> **需要**：rclone + Google Drive OAuth token（需在有浏览器的本地机器上授权一次）。

### 4.1 安装 rclone

```bash
sudo -v && curl https://rclone.org/install.sh | sudo bash
```

### 4.2 配置 Google Drive remote

**方法**：在本地有浏览器的机器上运行 `rclone authorize "drive"`，完成 Google 账号授权后，终端会输出一段 JSON token，将其填入服务器的配置文件：

```bash
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << 'EOF'
[gdrive]
type = drive
scope = drive
token = <粘贴从本地 rclone authorize "drive" 获取的 JSON token>
EOF
```

验证连接：

```bash
rclone lsd gdrive:dissert_ntu/models/
# 应能看到 google、runs_og 等目录
```

### 4.3 下载模型文件

Google Drive 路径：`dissert_ntu/models/`，包含两个目标目录：
- `google/` — PaliGemma-3B 基础模型（~10.9 GB）
- `runs_og/` — 预训练 checkpoint（~12.1 GB，含 `pick_and_place_ckpt/model_100000.pt`）

```bash
mkdir -p /workspace/models

# 并行下载（--transfers 8 并发，--drive-chunk-size 128M 加速大文件）
rclone copy gdrive:dissert_ntu/models/google /workspace/models/google \
    --drive-chunk-size 128M --transfers 8 --progress &

rclone copy gdrive:dissert_ntu/models/runs_og /workspace/models/runs_og \
    --drive-chunk-size 128M --transfers 8 --progress &

wait
echo "Downloads complete"
```

### 4.4 将 google 模型目录拷贝到仓库根目录

训练配置文件（`ft_libero_pi_object.yml`）中 `pretrained_model_path` 指向 `/workspace/galaxea_0/google/paligemma-3b-pt-224`，因此需要：

```bash
cp -r /workspace/models/google /workspace/galaxea_0/google
```

验证：

```bash
ls /workspace/galaxea_0/google/paligemma-3b-pt-224/
# 应能看到 config.json、model-*.safetensors 等文件
```

---

## 5. 第四步：下载训练数据集（HuggingFace）

数据集：[openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)，使用 `libero_object_no_noops` 子集（~2.82 GB，32 个 tfrecord 文件）。

```bash
mkdir -p /workspace/data

HF_TOKEN=<your_hf_token> huggingface-cli download openvla/modified_libero_rlds \
    --repo-type dataset \
    --include "libero_object_no_noops/*" \
    --local-dir /workspace/data
```

下载完成后目录结构：

```
/workspace/data/libero_object_no_noops/1.0.0/
├── dataset_info.json
├── features.json
├── dataset_statistics_*.json   # 首次训练时自动生成
└── libero_object-train.tfrecord-00000-of-00032
    ... (共 32 个 tfrecord)
```

---

## 6. 第五步：修改训练配置并启动 Finetune

### 6.1 关键配置文件

`/workspace/galaxea_0/vla/config/allen/libero_ft/ft_libero_pi_object.yml`

需要确认/修改的字段：

| 字段 | 说明 | 当前值 |
|------|------|--------|
| `vla_path` | 基础模型目录名（相对于仓库根目录） | `paligemma-3b-pt-224` |
| `MODEL.pretrained_model_path` | PaliGemma 模型绝对路径 | `/workspace/galaxea_0/google/paligemma-3b-pt-224` 需确认 |
| `data_root_dir` | 数据集根目录 | `/workspace/data` |
| `dataset_name` | 数据集名称 | `libero_object_no_noops` |
| `run_root_dir` | checkpoint 输出目录 | `runs` |
| `batch_size` | 每 GPU batch size | `12` |
| `grad_accumulation_steps` | 梯度累积步数（effective batch=384） | `32` |
| `max_steps` | 最大训练步数 | `15000` |
| `wandb_project` | WandB 项目名 | `libero_g0_og` |
| `wandb_entity` | WandB 实体名 | 见配置文件 |

> **注意**：配置文件中 `pretrained_model_path` 原来写的是 `/EFM-Pretrain/yuanty/cache/paligemma-3b-pt-224`（集群路径），在本机需要改为 `/workspace/galaxea_0/google/paligemma-3b-pt-224`。

### 6.2 启动 Finetune 训练

使用 2 个 GPU（`--nproc-per-node 2`）：

```bash
conda activate py10_g0_train
cd /workspace/galaxea_0

PYTHONPATH=/workspace/galaxea_0:$PYTHONPATH \
torchrun --standalone --nnodes 1 --nproc-per-node 2 finetune.py \
    --config vla/config/allen/libero_ft/ft_libero_pi_object.yml
```

训练输出：
- Checkpoint 保存在 `runs/ft_libero_pi_object--lora16_train1/`（按 `save_steps=100` 保存）
- WandB 日志实时上传
- 训练进度约 `43s/step`，15000 步总计约 180 小时（需多 GPU 或调低 max_steps）

### 6.3 使用已有 checkpoint 继续训练或直接评估

已有 checkpoint（从 Google Drive 下载的 `runs_og/pick_and_place_ckpt/model_100000.pt`）可以直接用于评估，无需重新训练。

---

## 7. 第六步：运行 Libero 评估

### 7.1 前置依赖（评估环境额外需要）

```bash
conda activate py10_g0_train

# 将 libero 仿真环境源码加入 Python 路径
echo "/workspace/openpi/third_party/libero" \
    > /venv/py10_g0_train/lib/python3.10/site-packages/libero_path.pth

# 安装评估所需额外依赖
pip install "robosuite==1.4.1"
pip install "bddl==1.0.1" "gym==0.25.2"
pip install future easydict imageio "imageio[ffmpeg]"
```

### 7.2 评估命令

```bash
conda activate py10_g0_train
cd /workspace/galaxea_0

# 首次运行时，libero 会询问数据集路径，输入 N 使用默认配置
echo "N" | PYTHONPATH=/workspace/galaxea_0:$PYTHONPATH \
python experiments/robot/libero/run_libero_eval.py \
    --config vla/config/allen/libero_ft/ft_libero_pi_object.yml \
    EVALUATION.pretrained_checkpoint=runs/ft_libero_pi_object--train1/model_ema_1600.pt \
    EVALUATION.task_suite_name=libero_object \
    EVALUATION.num_trials_per_task=10 \
    EVALUATION.output_dir=./evaluate_results/my_eval
```

评估结果：
- 视频：`./evaluate_results/my_eval/libero_object/videos/`
- 日志：`./evaluate_results/my_eval/EVAL-*.txt`
- 每个 episode 约 36 秒，100 个 episode 总计约 60 分钟

---

## 8. 评估环境已知问题与解决方案

### 问题 1：`ModuleNotFoundError: No module named 'libero'`

**原因**：`libero` 包位于 `/workspace/openpi/third_party/libero/`，`pip install -e` 因 `find_packages()` 缺少 `__init__.py` 失效。

**解决**：通过 `.pth` 文件加入路径（见 7.1）。

---

### 问题 2：`ModuleNotFoundError: No module named 'experiments'`

**原因**：脚本使用绝对导入，需要 `/workspace/galaxea_0` 在 `PYTHONPATH` 中。

**解决**：运行时加 `PYTHONPATH=/workspace/galaxea_0:$PYTHONPATH`。

---

### 问题 3：`ModuleNotFoundError: No module named 'imageio'`

```bash
pip install imageio
```

---

### 问题 4：`ModuleNotFoundError: No module named 'robosuite'` + API 不兼容

**原因**：最新版 robosuite 1.5.2 删除了 `robosuite.environments.manipulation.single_arm_env`，libero 依赖该模块。

**解决**：

```bash
pip install "robosuite==1.4.1"
```

---

### 问题 5：`ModuleNotFoundError: No module named 'bddl'`

```bash
pip install "bddl==1.0.1" "gym==0.25.2"
```

---

### 问题 6：`ModuleNotFoundError: No module named 'future'`

```bash
pip install future easydict einops
```

---

### 问题 7：`TypeError: get_libero_image() missing 1 required positional argument: 'resize_size'`

**原因**：`run_libero_eval_single.py` 第 190-191 行调用 `get_libero_image` 时漏传了 `env` 参数。

函数签名：`def get_libero_image(env, obs, resize_size)`（位于 `libero_utils.py:58`）

**错误代码**：
```python
img_ctx = get_libero_image(obs, 256)        # 错误：缺少 env
img = get_libero_image(obs, resize_size)    # 错误：缺少 env
```

**修复**：
```python
img_ctx = get_libero_image(env, obs, 256)
img = get_libero_image(env, obs, resize_size)
```

此 bug 已在本仓库中修复。

---

### 问题 8：`ValueError: Could not find a backend to open *.mp4 with iomode 'w?'`

**原因**：imageio 缺少 ffmpeg 后端。

**解决**：

```bash
pip install "imageio[ffmpeg]"
```

---

## 附：一键完整配置脚本

以下脚本假设服务器已安装 miniforge，有 2 张 GPU，CUDA 12.8：

```bash
#!/bin/bash
set -e

# ====== 变量（使用前修改） ======
PAT="ghp_xxxxxx"                        # GitHub PAT
GDRIVE_TOKEN='{"access_token":"...","refresh_token":"...","expiry":"..."}'  # rclone token
HF_TOKEN="hf_xxxxxx"                    # HuggingFace token
# ================================

# 1. Clone repo
cd /workspace
git clone https://${PAT}@github.com/CharlieXie/galaxea_0.git
cd galaxea_0 && git checkout main

# 2. Conda env
conda env create -f env.yaml
git clone https://github.com/kvablack/dlimp.git /workspace/dlimp
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate py10_g0_train
pip install -e /workspace/dlimp
pip install -e /workspace/galaxea_0/openpi-client

# 3. rclone + Google Drive
sudo -v && curl https://rclone.org/install.sh | sudo bash
mkdir -p ~/.config/rclone
echo "[gdrive]
type = drive
scope = drive
token = ${GDRIVE_TOKEN}" > ~/.config/rclone/rclone.conf

mkdir -p /workspace/models
rclone copy gdrive:dissert_ntu/models/google /workspace/models/google \
    --drive-chunk-size 128M --transfers 8 --progress &
rclone copy gdrive:dissert_ntu/models/runs_og /workspace/models/runs_og \
    --drive-chunk-size 128M --transfers 8 --progress &
wait

cp -r /workspace/models/google /workspace/galaxea_0/google

# 4. HuggingFace dataset
mkdir -p /workspace/data
HF_TOKEN=${HF_TOKEN} huggingface-cli download openvla/modified_libero_rlds \
    --repo-type dataset \
    --include "libero_object_no_noops/*" \
    --local-dir /workspace/data

# 5. Eval dependencies
echo "/workspace/openpi/third_party/libero" \
    > /venv/py10_g0_train/lib/python3.10/site-packages/libero_path.pth
pip install "robosuite==1.4.1" "bddl==1.0.1" "gym==0.25.2" \
    future easydict imageio "imageio[ffmpeg]"

echo "Setup complete!"
```

---

## 注意事项

- **conda 环境路径**：`py10_g0_train` 实际在 `/venv/py10_g0_train`，用 `conda activate py10_g0_train` 或直接用 `/venv/py10_g0_train/bin/python`。
- **模型路径**：配置文件中 `pretrained_model_path` 原为集群路径 `/EFM-Pretrain/...`，需确保改为 `/workspace/galaxea_0/google/paligemma-3b-pt-224`。
- **数据集警告**：评估时 `[Warning]: datasets path ... does not exist!` 是 libero 默认路径不存在的无害警告。
- **EGL 警告**：结束时 `OpenGL.raw.EGL._errors.EGLError` 是 robosuite 渲染上下文清理时的无害警告。
- **训练速度**：2 GPU 下约 43s/step，15000 步约需 180 小时，建议根据实际需求调整 `max_steps`。
