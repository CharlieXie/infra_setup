# V1 E2E 训练环境 Runbook

> 目标读者：AI Agent。从零搭建 Joint（VLM+AE 共享 backbone）训练环境。
>
> 最后验证：2026-03-29，Ubuntu 24.04，CUDA 12.8，2× RTX PRO 6000 Blackwell (~95 GB)

---

## Agent 行为准则

1. **轮询长任务用循环**：`while ! grep -q "完成标志" log; do sleep 10 && tail -2 log; done`。
2. **后台任务用 `&`**，输出重定向文件，用 `tail` 检查。
3. **发现错误立刻读日志**，不要盲目重试。
4. **stdin 交互用管道传入**：`echo "N" | command`，避免挂起。

---

## 1. 克隆代码

```bash
cd /workspace
GIT_LFS_SKIP_SMUDGE=1 git clone https://<PAT>@github.com/CharlieXie/openpi.git
cd openpi
git checkout v1.2_e2e
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
git branch  # 验证: * v1.2_e2e
```

---

## 2. Python 环境

```bash
cd /workspace/openpi

# 2.1 uv sync（后台，3-10 分钟）
GIT_LFS_SKIP_SMUDGE=1 uv sync > /tmp/uv_sync.log 2>&1 &
while ! grep -q "Installed" /tmp/uv_sync.log 2>/dev/null; do sleep 10 && tail -2 /tmp/uv_sync.log; done
echo "uv sync done"

# 2.2 TF（后台，1-2 分钟）
uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3" > /tmp/tf.log 2>&1 &
while ! grep -q "Installed" /tmp/tf.log 2>/dev/null; do sleep 10 && tail -2 /tmp/tf.log; done
.venv/bin/python -c "import tensorflow as tf; print('TF:', tf.__version__)"  # 2.15.0

# 2.3 升级 NCCL（PyTorch 自带 2.26.2 在 Blackwell 多卡有 bug）
uv pip install --python .venv/bin/python "nvidia-nccl-cu12>=2.29"

# 2.4 transformers patch
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
.venv/bin/python -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print('patch OK')
"
```

> **venv 加速**：重复部署时可备份 `.venv/` 跳过 uv sync：
> `tar -czf /workspace/venv_backup.tar.gz -C /workspace/openpi .venv`
> 还原：`tar -xzf /workspace/venv_backup.tar.gz -C /workspace/openpi`

---

## 3. 数据准备

> ⚠️ `/workspace/data/` 可能是只读 bind mount，`dataset_statistics.json` 须保存到可写目录。

### 生成 dataset_statistics.json

如果已有统计文件且格式正确（flat 格式，action=7 维，proprio=6 维），跳过本节。

**LIBERO（约 60s）**：
```bash
cd /workspace/openpi && mkdir -p data
.venv/bin/python scripts/compute_wp_norm_stats.py \
    --rlds_dir /workspace/data/object/libero_object_no_noops/libero_object_no_noops/1.0.0 \
    --robot_type libero \
    --output_dir /workspace/openpi/data
# 验证: action dims=7, proprio dims=6
```

**CALVIN ABC_D**：
```bash
.venv/bin/python scripts/compute_wp_norm_stats.py \
    --rlds_dir /workspace/calvin_abc_wp_001/calvin_abc_wp/1.0.0 \
    --robot_type calvin \
    --output_dir /workspace/calvin_abc_wp_001/norm_stats
```

验证方法（通用）：
```bash
.venv/bin/python -c "
import json, sys; d = json.load(open(sys.argv[1]))
print('action:', len(d['action']['q99']), 'proprio:', len(d['proprio']['q99']), 'steps:', d['num_transitions'])
" <path_to_dataset_statistics.json>
# 期望: action: 7  proprio: 6
```

---

## 4. Wandb 配置

```bash
# 新版 key (wandb_v1_...) 必须用 netrc，不能 wandb login
echo "machine api.wandb.ai
  login user
  password <your_wandb_api_key>" >> ~/.netrc
chmod 600 ~/.netrc
```

| 项目 | 值 |
|------|----|
| Project | `waypoint_e2e` |
| Entity | `chuanliang-xie-nanyang-technological-university-singapore` |

---

## 5. 启动训练

### LIBERO

```bash
cd /workspace/openpi && mkdir -p logs

tmux kill-session -t joint_train 2>/dev/null; sleep 1
tmux new-session -d -s joint_train -x 220 -y 50
tmux send-keys -t joint_train "cd /workspace/openpi" Enter; sleep 2
tmux send-keys -t joint_train "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter; sleep 2
tmux send-keys -t joint_train "export WANDB_API_KEY=<your_wandb_api_key>" Enter; sleep 2
tmux send-keys -t joint_train ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint_joint.py --config configs/waypoint_joint_libero.yaml 2>&1 | tee logs/waypoint_joint_libero_object.log" Enter
```

配置文件 `configs/waypoint_joint_libero.yaml` 关键参数：

| 参数 | 值 | 参数 | 值 |
|------|----|------|----|
| `pretrained_weight_path` | `/workspace/models/pi05_base_pytorch` | `vlm_batch_size` / `ae_batch_size` | `24` (per GPU) |
| `dataset_statistics_path` | `/workspace/openpi/data` | `num_train_steps` | `10000` |
| `gradient_strategy` | `stop_gradient` | `peak_lr` | `5.0e-5` |
| `lora_enabled` | `false` | `save_interval` | `800` |

期望初始 loss：VLM 11–12，AE 0.7–1.0。

### CALVIN ABC_D

```bash
cd /workspace/openpi && mkdir -p logs

tmux kill-session -t joint_train 2>/dev/null; sleep 1
tmux new-session -d -s joint_train -x 220 -y 50
tmux send-keys -t joint_train "cd /workspace/openpi" Enter; sleep 2
tmux send-keys -t joint_train "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter; sleep 2
tmux send-keys -t joint_train "export WANDB_API_KEY=<your_wandb_api_key>" Enter; sleep 2
tmux send-keys -t joint_train ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint_joint.py --config configs/waypoint_joint_calvin.yaml 2>&1 | tee logs/waypoint_joint_calvin.log" Enter
```

配置文件 `configs/waypoint_joint_calvin.yaml` 关键参数：

| 参数 | 值 | 参数 | 值 |
|------|----|------|----|
| `pretrained_weight_path` | `/workspace/models/600` | `vlm_batch_size` / `ae_batch_size` | `192` (per GPU) |
| `dataset_statistics_path` | `/workspace/data/dataset_statistics.json` | `num_train_steps` | `5000` |
| `gradient_strategy` | `scale_gradient` (0.1) | `peak_lr` | `1.0e-4` |
| `lora_enabled` | `false` | `save_interval` | `200` |

期望初始 loss：VLM 2–3，AE 0.1–0.2，显存 ~27 GB / 95 GB。

### 监控

```bash
tail -f logs/waypoint_joint_libero_object.log | grep "\[Joint\]"   # LIBERO
tail -f logs/waypoint_joint_calvin.log | grep "\[Joint\]"          # CALVIN
tmux attach -t joint_train                                          # Ctrl+B, D 退出
```

---

## 6. LoRA 训练与 Merge

### 何时需要

| 训练方式 | `lora_enabled` | checkpoint 内容 | 评测前操作 |
|----------|---------------|-----------------|-----------|
| 全量微调 | `false`（当前默认） | `model.safetensors`（完整权重） | 无需处理 |
| LoRA 微调 | `true` | `lora.safetensors`（delta 权重） | **必须 merge** |

### LoRA 配置参数

```yaml
lora_enabled: true
lora_rank: 16
lora_alpha: 16.0
lora_dropout: 0.0
lora_apply_to: "all"
```

### Merge 命令

```bash
cd /workspace/openpi
.venv/bin/python scripts/merge_lora.py \
    --base <base_model.safetensors> \
    --lora <checkpoint_dir>/<step>/lora.safetensors \
    --config <training_config.yaml> \
    --output <checkpoint_dir>/<step>/model_merged.safetensors
```

LIBERO 示例：
```bash
.venv/bin/python scripts/merge_lora.py \
    --base /workspace/models/pi05_base_pytorch/model.safetensors \
    --lora checkpoints/waypoint_joint_libero_sg/600/lora.safetensors \
    --config configs/waypoint_joint_libero.yaml \
    --output checkpoints/waypoint_joint_libero_sg/600/model_merged.safetensors
```

Merge 后，评测 config 的 `joint_checkpoint` 指向该 `<step>/` 目录即可。eval 自动优先加载 `model_merged.safetensors`，不存在时才加载 `model.safetensors`。

---

## 关键路径

| 资源 | LIBERO | CALVIN |
|------|--------|--------|
| 训练配置 | `configs/waypoint_joint_libero.yaml` | `configs/waypoint_joint_calvin.yaml` |
| 训练日志 | `logs/waypoint_joint_libero_object.log` | `logs/waypoint_joint_calvin.log` |
| Checkpoint | `checkpoints/waypoint_joint_libero_sg/` | `checkpoints/waypoint_joint_calvin_sg_0.1_t1/` |
| RLDS 数据 | `/workspace/data/object/libero_object_no_noops/.../1.0.0` | `/workspace/data/calvin_abc_rlds` |
| Waypoint RLDS | `/workspace/data/object/libero_object_wp_001/.../1.0.0` | `/workspace/data/calvin_abc_wp_0_02/calvin_abc_wp/1.0.0` |
| Dataset stats | `/workspace/openpi/data/dataset_statistics.json` | `/workspace/data/dataset_statistics.json` |
| Base 权重 | `/workspace/models/pi05_base_pytorch/model.safetensors` | `/workspace/models/600/model.safetensors` |
