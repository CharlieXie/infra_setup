# OpenPI Training Quickstart

> 最后验证：2026-02-25，硬件：2× RTX PRO 6000 Blackwell，Ubuntu 24.04，CUDA 12.8

---

## 1. 基础环境

```bash
touch ~/.no_auto_tmux

git config --global user.email "chuanliang.xie@gmail.com"
git config --global user.name "chuanliang"

sudo apt-get install -y ffmpeg pkg-config build-essential \
    libosmesa6-dev libgles2 libegl1
```

---

## 2. 克隆代码仓库

> **需要用户输入**：将 `<PAT>` 替换为 GitHub Personal Access Token（格式：`ghp_xxxx`）

```bash
cd /workspace
git clone https://<PAT>@github.com/CharlieXie/openpi.git
cd openpi
git checkout pytorch_lora_blackwell
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

---

## 3. Python 环境（uv sync）

```bash
# 确认 av override 存在（pytorch_lora_blackwell 分支已预置，通常无需修改）
grep "override-dependencies" /workspace/openpi/pyproject.toml
# 期望输出包含: "av>=13.1.0,<14.0.0"

cd /workspace/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync > /tmp/uv_sync.log 2>&1 &
echo "PID=$!"
```

监控：`sleep 30 && tail -10 /tmp/uv_sync.log`（重复直到出现 `Installed N packages`）

完成后安装 TF 并打 transformers patch：

```bash
cd /workspace/openpi

# TF 必须 2.15.0，高版本与 ml_dtypes 冲突
uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"

cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/

.venv/bin/python -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print('patch OK')
"
```

---

## 4. 生成 dataset_statistics

### 4.1 AE stats（从完整 RLDS，约 60s）

> ⚠️ Google Drive 上的 stats 维度不匹配，必须从 RLDS 重算。

```bash
cd /workspace/openpi
.venv/bin/python - << 'PYEOF' > /tmp/ae_stats.log 2>&1 &
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import numpy as np, json

b = tfds.builder_from_directory('/workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0')
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
path = '/workspace/data/dataset_statistics.json'
with open(path, 'w') as f: json.dump(out, f, indent=2)
print('Saved to', path)
PYEOF
echo "AE stats PID=$!"
```

监控：`sleep 30 && tail -5 /tmp/ae_stats.log`

### 4.2 VLM stats（从 waypoint-filtered RLDS，约 30-40s）

> **需要用户确认路径**：
> - `--rlds_dir`：waypoint-filtered RLDS 的 `1.0.0` 目录路径
> - `--output_dir`：stats 输出目录

```bash
cd /workspace/openpi
.venv/bin/python scripts/compute_wp_norm_stats.py \
    --rlds_dir /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0 \
    --robot_type libero \
    --output_dir /workspace/data/libero/libero_object_wp_001/norm_stats
```

验证（完成后）：
```bash
.venv/bin/python -c "
import json; d = json.load(open('/workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json'))
print('action:', len(d['action']['q99']), 'proprio:', len(d['proprio']['q99']), 'steps:', d['num_transitions'])
# 期望: 7  8  8863
"
```

---

## 5. 配置 wandb

> **需要用户输入**：将 `<your_wandb_api_key>` 替换为实际的 wandb API key
>
> 注意：新版 key（`wandb_v1_...`，86字符）不能用 `wandb login`，必须用 netrc 方式。

```bash
echo "machine api.wandb.ai
  login user
  password <your_wandb_api_key>" >> ~/.netrc
chmod 600 ~/.netrc
```

---

## 6. 确认所有路径

```bash
ls /workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/dataset_info.json && echo "✓ RLDS"
ls /workspace/data/libero/libero_object_wp_001/waypoint_indices.json && echo "✓ wp_indices"
ls /workspace/data/dataset_statistics.json && echo "✓ AE stats"
ls /workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json && echo "✓ VLM stats"
ls /workspace/models/pi05_base_pytorch/model.safetensors && echo "✓ model"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

---

## 7. 启动训练

> **需要用户输入**：将两处 `<your_wandb_key>` 替换为实际的 wandb API key
>
> tmux 规则：每条 `send-keys` 只发一条命令，间隔 `sleep 2`，避免 bash 解析混乱。

### 7.1 AE 训练

```bash
tmux kill-session -t waypoint_ae 2>/dev/null; sleep 1
tmux new-session -d -s waypoint_ae -x 220 -y 50

tmux send-keys -t waypoint_ae "cd /workspace/openpi" Enter; sleep 2
tmux send-keys -t waypoint_ae "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter; sleep 2
tmux send-keys -t waypoint_ae "export WANDB_API_KEY=<your_wandb_key>" Enter; sleep 2

mkdir -p /workspace/openpi/logs
tmux send-keys -t waypoint_ae ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint.py --mode ae --config configs/waypoint_ae_libero.yaml 2>&1 | tee logs/waypoint_ae_libero.log" Enter
```

验证（30s 后）：
```bash
sleep 30 && tail -20 /workspace/openpi/logs/waypoint_ae_libero.log
```
期望出现：`[AE] step=0/10000 loss=0.xxx`（初始 loss 0.7-1.0）

### 7.2 VLM 训练

```bash
tmux kill-session -t waypoint_vlm 2>/dev/null; sleep 1
tmux new-session -d -s waypoint_vlm -x 220 -y 50

tmux send-keys -t waypoint_vlm "cd /workspace/openpi" Enter; sleep 2
tmux send-keys -t waypoint_vlm "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter; sleep 2
tmux send-keys -t waypoint_vlm "export WANDB_API_KEY=<your_wandb_key>" Enter; sleep 2

tmux send-keys -t waypoint_vlm ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint.py --mode vlm --config configs/waypoint_vlm_libero.yaml 2>&1 | tee logs/waypoint_vlm_libero.log" Enter
```

验证（30s 后）：
```bash
sleep 30 && tail -20 /workspace/openpi/logs/waypoint_vlm_libero.log
```
期望出现：`[VLM] step=0/30000 loss=11.xxx`（初始 CE loss 11-12）

---

## 快速监控

```bash
# 实时追踪训练日志
tail -f /workspace/openpi/logs/waypoint_ae_libero.log | grep "\[AE\]"
tail -f /workspace/openpi/logs/waypoint_vlm_libero.log | grep "\[VLM\]"

# GPU 使用率
watch -n 5 nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader

# 进入 tmux 查看详细输出（Ctrl+B, D 退出不杀进程）
tmux attach -t waypoint_ae
tmux attach -t waypoint_vlm
```

---

## 关键路径速查

| 资源 | 路径 |
|------|------|
| AE 训练配置 | `/workspace/openpi/configs/waypoint_ae_libero.yaml` |
| VLM 训练配置 | `/workspace/openpi/configs/waypoint_vlm_libero.yaml` |
| Pi0.5 PyTorch 权重 | `/workspace/models/pi05_base_pytorch/model.safetensors` |
| LIBERO RLDS | `/workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/` |
| Waypoint indices | `/workspace/data/libero/libero_object_wp_001/waypoint_indices.json` |
| Waypoint filtered RLDS | `/workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0/` |
| AE dataset statistics | `/workspace/data/dataset_statistics.json` |
| VLM dataset statistics | `/workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json` |
| AE 训练日志 | `/workspace/openpi/logs/waypoint_ae_libero.log` |
| VLM 训练日志 | `/workspace/openpi/logs/waypoint_vlm_libero.log` |
| Google Drive | `gg1:dissert_ntu/libero/` |
