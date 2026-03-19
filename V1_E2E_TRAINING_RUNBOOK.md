# E2E 环境配置记录

> 最后验证：2026-03-18，硬件：1× RTX PRO 6000 Blackwell，Ubuntu 24.04，CUDA 12.8

---

## 硬件 & 系统

| 项目 | 值 |
|------|----|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| 显存 | 97,887 MiB (~95 GB) |
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
git checkout v1_e2e
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

| 项目 | 值 |
|------|----|
| 仓库路径 | `/workspace/openpi` |
| 分支 | `v1_e2e` |
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

### 安装步骤

```bash
cd /workspace/openpi

# 1. 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 2. 安装 TF（必须 2.15.0）
uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"

# 3. 打 transformers patch
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/

# 4. 验证 patch
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

### 生成 dataset_statistics.json

```bash
cd /workspace/openpi && mkdir -p data

.venv/bin/python - << 'PYEOF'
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import numpy as np, json

b = tfds.builder_from_directory('/workspace/data/object/libero_object_no_noops/libero_object_no_noops/1.0.0')
ds = b.as_dataset(split='train')
all_actions, all_proprios = [], []
for ep in ds:
    for step in ep['steps']:
        all_actions.append(step['action'].numpy().astype('float32'))
        all_proprios.append(step['observation']['state'].numpy().astype('float32').flatten())
all_actions = np.stack(all_actions); all_proprios = np.stack(all_proprios)
print(f'Actions: {all_actions.shape}, Proprios: {all_proprios.shape}')  # (66984,7), (66984,8)
def stats(arr):
    return {'mean': arr.mean(0).tolist(), 'std': arr.std(0).tolist(),
            'q01': np.percentile(arr,1,0).tolist(), 'q99': np.percentile(arr,99,0).tolist(),
            'min': arr.min(0).tolist(), 'max': arr.max(0).tolist()}
out = {'libero_object_no_noops': {'action': stats(all_actions), 'proprio': stats(all_proprios), 'num_samples': len(all_actions)}}
with open('/workspace/openpi/data/dataset_statistics.json', 'w') as f: json.dump(out, f, indent=2)
print('Saved.')
PYEOF
```

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

### 训练命令

```bash
cd /workspace/openpi
mkdir -p logs

# 启动 tmux session
tmux kill-session -t joint_train 2>/dev/null; sleep 1
tmux new-session -d -s joint_train -x 220 -y 50

tmux send-keys -t joint_train "cd /workspace/openpi" Enter; sleep 2
tmux send-keys -t joint_train "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter; sleep 2
tmux send-keys -t joint_train "export WANDB_API_KEY=<your_wandb_api_key>" Enter; sleep 2
tmux send-keys -t joint_train ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_waypoint_joint.py --config configs/waypoint_joint_libero.yaml 2>&1 | tee logs/waypoint_joint_libero_object.log" Enter
```

> ⚠️ 本机只有 1 块 GPU，使用 `--nproc_per_node=1`（原始文档为 2）。

### 关键配置文件：`configs/waypoint_joint_libero.yaml`

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
| `num_train_steps` | `20000` |
| `warmup_steps` | `400` |
| `peak_lr` | `9.0e-5` |
| `end_lr` | `1.0e-7` |
| `gradient_strategy` | `none` |
| `ae_loss_weight` | `1.0` |
| `lora_enabled` | `false` |
| `train_vision_encoder` | `true` |
| `wandb_project` | `waypoint_e2e` |
| `exp_name` | `waypoint_joint_libero_01` |
| `checkpoint_dir` | `checkpoints/waypoint_joint_libero_01` |
| `save_interval` | `1500` |

### 期望初始 loss

| 指标 | 期望范围 |
|------|---------|
| VLM loss (step=0) | 11–12 |
| AE loss (step=0) | 0.7–1.0 |

---

## 快速监控

```bash
# 实时训练日志
tail -f /workspace/openpi/logs/waypoint_joint_libero_object.log | grep "\[Joint\]"

# GPU 使用率
watch -n 5 nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader

# 进入 tmux（Ctrl+B, D 退出不杀进程）
tmux attach -t joint_train
```

---

## 关键路径速查

| 资源 | 路径 |
|------|------|
| 训练脚本 | `/workspace/openpi/scripts/train_waypoint_joint.py` |
| 训练配置 | `/workspace/openpi/configs/waypoint_joint_libero.yaml` |
| 训练日志 | `/workspace/openpi/logs/waypoint_joint_libero_object.log` |
| Checkpoint | `/workspace/openpi/checkpoints/waypoint_joint_libero_01/` |
| Dataset statistics | `/workspace/openpi/data/dataset_statistics.json` |
| Pi0.5 base 权重 | `/workspace/models/pi05_base_pytorch/model.safetensors` |
| LIBERO RLDS | `/workspace/data/object/libero_object_no_noops/libero_object_no_noops/1.0.0/` |
| Waypoint filtered RLDS | `/workspace/data/object/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0/` |
