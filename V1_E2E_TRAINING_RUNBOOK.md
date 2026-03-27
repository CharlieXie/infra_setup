# E2E 环境配置记录

> 最后验证：2026-03-19，硬件：2× RTX PRO 6000 Blackwell，Ubuntu 24.04，CUDA 12.8

---

## 硬件 & 系统

| 项目 | 值 |
|------|----|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition × 2 |
| 显存 | 97,887 MiB (~95 GB) × 2 |
| CUDA 版本 | 12.8 |
| 驱动版本 | 590.48.01 |
| OS | Ubuntu 24.04 |

---

## 代码仓库

```bash
# openpi — 训练代码
cd /workspace
git clone https://<PAT>@github.com/CharlieXie/openpi.git
cd openpi
git checkout v1.2_e2e
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

| 项目 | 值 |
|------|----|
| 仓库路径 | `/workspace/openpi` |
| 分支 | `v1.2_e2e` |
| 子模块 | `third_party/aloha`, `third_party/libero` |

---

## Python 环境

| 项目 | 值 |
|------|----|
| 包管理器 | uv 0.9.21 |
| Python | 3.11.14 |
| venv 路径 | `/workspace/openpi/.venv` |
| PyTorch | 2.7.1+cu128 |
| TensorFlow | 2.15.0 |
| tensorflow-datasets | 4.9.3 |
| transformers | 4.53.2（已打 siglip patch） |
| nvidia-nccl-cu12 | ≥2.29（Blackwell 多卡必须升级，PyTorch 自带 2.26.2 有 bug） |

### 安装步骤

```bash
cd /workspace/openpi

# 1. 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 2. 安装 TF（必须 2.15.0）
uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"

# 3. 升级 NCCL（PyTorch 自带的 2.26.2 在 Blackwell 多卡通信时有 bug）
uv pip install --python .venv/bin/python "nvidia-nccl-cu12>=2.29"

# 4. 打 transformers patch
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/

# 5. 验证 patch
.venv/bin/python -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print('patch OK')
"
```

---

## 数据路径

> ⚠️ `/workspace/data/` 是只读 bind mount（owner: nobody），不可写入。
> `dataset_statistics.json` 须保存到 `/workspace/openpi/data/` 目录。

| 资源 | 路径 |
|------|------|
| LIBERO RLDS (full) | `/workspace/data/object/libero_object_no_noops/libero_object_no_noops/1.0.0` |
| Waypoint indices | `/workspace/data/object/libero_object_wp_001/waypoint_indices.json` |
| Waypoint filtered RLDS | `/workspace/data/object/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0` |
| Dataset statistics | `/workspace/openpi/data/dataset_statistics.json` |
| Pi0.5 base 权重 | `/workspace/models/pi05_base_pytorch/model.safetensors` |

### 准备 dataset_statistics.json

使用 `compute_wp_norm_stats.py` 从 waypoint-filtered RLDS 数据计算并保存到可写目录。脚本通过 `--robot_type` 自动适配不同机器人的 gripper 归一化和 proprio 维度。

#### LIBERO（约 60s）

```bash
cd /workspace/openpi && mkdir -p data
.venv/bin/python scripts/compute_wp_norm_stats.py \
    --rlds_dir /workspace/data/object/libero_object_no_noops/libero_object_no_noops/1.0.0 \
    --robot_type libero \
    --output_dir /workspace/openpi/data
# 完成标志: "Saved to /workspace/openpi/data/dataset_statistics.json"
```

验证：
```bash
.venv/bin/python -c "
import json; d = json.load(open('/workspace/openpi/data/dataset_statistics.json'))
print('action dims:', len(d['action']['q99']))   # 7
print('proprio dims:', len(d['proprio']['q99']))  # 6 (连续维度，不含 gripper)
print('steps:', d['num_transitions'])
"
```

#### CALVIN ABC_D

```bash
cd /workspace/openpi
.venv/bin/python scripts/compute_wp_norm_stats.py \
    --rlds_dir /workspace/calvin_abc_wp_001/calvin_abc_wp/1.0.0 \
    --robot_type calvin \
    --output_dir /workspace/calvin_abc_wp_001/norm_stats
# 完成标志: "Saved to /workspace/calvin_abc_wp_001/norm_stats/dataset_statistics.json"
```

验证：
```bash
.venv/bin/python -c "
import json; d = json.load(open('/workspace/calvin_abc_wp_001/norm_stats/dataset_statistics.json'))
print('action dims:', len(d['action']['q99']))   # 7 (delta_pos(3) + delta_euler(3) + gripper(1))
print('proprio dims:', len(d['proprio']['q99']))  # 6 (TCP pos(3) + euler(3), 不含 gripper width)
print('steps:', d['num_transitions'])
"
```

> **说明**：脚本对 action gripper 维度应用 `normalize_gripper`（LIBERO: clip+invert; CALVIN: clip only），proprio 只保留连续维度（dims 0-5），输出 flat JSON（无外层 dataset 名称 key）。同一个脚本通过 `--robot_type` 参数自动适配不同机器人。

---

## Wandb 配置

> 新版 key（`wandb_v1_...`）不能用 `wandb login`，必须用 netrc。

```bash
echo "machine api.wandb.ai
  login user
  password <your_wandb_api_key>" >> ~/.netrc
chmod 600 ~/.netrc
```

| 项目 | 值 |
|------|----|
| Project | `waypoint_e2e` |
| Entity | `chuanliang-xie-nanyang-technological-university-singapore` |
| Run URL 模板 | `https://wandb.ai/<entity>/waypoint_e2e/runs/<run_id>` |

---

## 训练配置（Joint VLM + AE）

### LIBERO

#### 训练命令

```bash
cd /workspace/openpi
mkdir -p logs

# 启动 tmux session
tmux kill-session -t joint_train 2>/dev/null; sleep 1
tmux new-session -d -s joint_train -x 220 -y 50

tmux send-keys -t joint_train "cd /workspace/openpi" Enter; sleep 2
tmux send-keys -t joint_train "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter; sleep 2
tmux send-keys -t joint_train "export WANDB_API_KEY=<your_wandb_api_key>" Enter; sleep 2
tmux send-keys -t joint_train ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint_joint.py --config configs/waypoint_joint_libero.yaml 2>&1 | tee logs/waypoint_joint_libero_object.log" Enter
```

#### 关键配置文件：`configs/waypoint_joint_libero.yaml`

| 参数 | 值 |
|------|----|
| `original_rlds_dir` | `/workspace/data/object/libero_object_no_noops/libero_object_no_noops/1.0.0` |
| `wp_indices_path` | `/workspace/data/object/libero_object_wp_001/waypoint_indices.json` |
| `wp_rlds_dir` | `/workspace/data/object/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0` |
| `dataset_statistics_path` | `/workspace/openpi/data` |
| `pretrained_weight_path` | `/workspace/models/pi05_base_pytorch` |
| `paligemma_variant` | `gemma_2b` |
| `action_expert_variant` | `gemma_300m` |
| `precision` | `bfloat16` |
| `vlm_batch_size` | `24` (per GPU) |
| `ae_batch_size` | `24` (per GPU) |
| `num_train_steps` | `10000` |
| `warmup_steps` | `400` |
| `peak_lr` | `5.0e-5` |
| `end_lr` | `1.0e-7` |
| `gradient_strategy` | `stop_gradient` |
| `ae_loss_weight` | `1.0` |
| `lora_enabled` | `false` |
| `train_vision_encoder` | `true` |
| `wandb_project` | `waypoint_e2e` |
| `exp_name` | `waypoint_joint_libero_sg` |
| `checkpoint_dir` | `checkpoints/{exp_name}` |
| `save_interval` | `800` |

#### 期望初始 loss

| 指标 | 期望范围 |
|------|---------|
| VLM loss (step=0) | 11–12 |
| AE loss (step=0) | 0.7–1.0 |

---

### CALVIN ABC_D

#### 数据路径

| 资源 | 路径 |
|------|------|
| CALVIN RLDS (full) | `/workspace/data/calvin_abc_rlds` |
| Waypoint indices | `/workspace/data/calvin_abc_wp_0_02/waypoint_indices.json` |
| Waypoint filtered RLDS | `/workspace/data/calvin_abc_wp_0_02/calvin_abc_wp/1.0.0` |
| Dataset statistics | `/workspace/data/dataset_statistics.json`（预置，可直接使用） |
| Pi0.5 base 权重 | `/workspace/models/600/model.safetensors` |

> `dataset_statistics.json` 已预置于 `/workspace/data/`，验证如下：
> action dim=7（delta_pos×3 + delta_euler×3 + gripper×1），proprio dim=6（TCP pos×3 + euler×3），num_transitions=171,250。

#### 训练命令

```bash
cd /workspace/openpi
mkdir -p logs

# 启动 tmux session
tmux kill-session -t joint_train 2>/dev/null; sleep 1
tmux new-session -d -s joint_train -x 220 -y 50

tmux send-keys -t joint_train "cd /workspace/openpi" Enter; sleep 2
tmux send-keys -t joint_train "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter; sleep 2
tmux send-keys -t joint_train "export WANDB_API_KEY=<your_wandb_api_key>" Enter; sleep 2
tmux send-keys -t joint_train ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint_joint.py --config configs/waypoint_joint_calvin.yaml 2>&1 | tee logs/waypoint_joint_calvin.log" Enter
```

#### 关键配置文件：`configs/waypoint_joint_calvin.yaml`

| 参数 | 值 |
|------|----|
| `original_rlds_dir` | `/workspace/data/calvin_abc_rlds` |
| `wp_indices_path` | `/workspace/data/calvin_abc_wp_0_02/waypoint_indices.json` |
| `wp_rlds_dir` | `/workspace/data/calvin_abc_wp_0_02/calvin_abc_wp/1.0.0` |
| `dataset_statistics_path` | `/workspace/data/dataset_statistics.json` |
| `pretrained_weight_path` | `/workspace/models/600` |
| `paligemma_variant` | `gemma_2b` |
| `action_expert_variant` | `gemma_300m` |
| `precision` | `bfloat16` |
| `vlm_batch_size` | `192` (per GPU) |
| `ae_batch_size` | `192` (per GPU) |
| `num_train_steps` | `5000` |
| `warmup_steps` | `100` |
| `peak_lr` | `1.0e-4` |
| `end_lr` | `1.0e-7` |
| `gradient_strategy` | `scale_gradient` |
| `gradient_scale` | `0.1` |
| `ae_loss_weight` | `1.0` |
| `lora_enabled` | `false` |
| `train_vision_encoder` | `true` |
| `wandb_project` | `waypoint_e2e` |
| `exp_name` | `waypoint_joint_calvin_sg_0.1_t1` |
| `checkpoint_dir` | `checkpoints/{exp_name}` |
| `save_interval` | `200` |

#### 期望初始 loss（实测 2026-03-27）

| 指标 | 期望范围 | 实测值 |
|------|---------|--------|
| VLM loss (step=0) | 2–3 | 2.26 |
| AE loss (step=0) | 0.1–0.2 | 0.15 |
| GPU 显存占用 | ~27 GB / 95 GB | 27.4 GB |

---

## 快速监控

```bash
# 实时训练日志（LIBERO）
tail -f /workspace/openpi/logs/waypoint_joint_libero_object.log | grep "\[Joint\]"

# 实时训练日志（CALVIN）
tail -f /workspace/openpi/logs/waypoint_joint_calvin.log | grep "\[Joint\]"

# GPU 使用率
watch -n 5 nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader

# 进入 tmux（Ctrl+B, D 退出不杀进程）
tmux attach -t joint_train
```

---

## 关键路径速查

### LIBERO

| 资源 | 路径 |
|------|------|
| 训练脚本 | `/workspace/openpi/scripts/train_waypoint_joint.py` |
| 训练配置 | `/workspace/openpi/configs/waypoint_joint_libero.yaml` |
| 训练日志 | `/workspace/openpi/logs/waypoint_joint_libero_object.log` |
| Checkpoint | `/workspace/openpi/checkpoints/waypoint_joint_libero_sg/` |
| Dataset statistics | `/workspace/openpi/data/dataset_statistics.json` |
| Pi0.5 base 权重 | `/workspace/models/pi05_base_pytorch/model.safetensors` |
| LIBERO RLDS | `/workspace/data/object/libero_object_no_noops/libero_object_no_noops/1.0.0/` |
| Waypoint filtered RLDS | `/workspace/data/object/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0/` |

### CALVIN ABC_D

| 资源 | 路径 |
|------|------|
| 训练脚本 | `/workspace/openpi/scripts/train_waypoint_joint.py` |
| 训练配置 | `/workspace/openpi/configs/waypoint_joint_calvin.yaml` |
| 训练日志 | `/workspace/openpi/logs/waypoint_joint_calvin.log` |
| Checkpoint | `/workspace/openpi/checkpoints/waypoint_joint_calvin_sg_0.1_t1/` |
| Dataset statistics | `/workspace/data/dataset_statistics.json` |
| Pi0.5 base 权重 | `/workspace/models/600/model.safetensors` |
| CALVIN RLDS | `/workspace/data/calvin_abc_rlds/` |
| Waypoint filtered RLDS | `/workspace/data/calvin_abc_wp_0_02/calvin_abc_wp/1.0.0/` |
