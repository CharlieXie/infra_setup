# OpenPI Waypoint VLA — 训练 Runbook（AE + VLM）

> **目标读者**: AI Agent。在全新 vast.ai 服务器上从零启动 Action Expert (AE) 和 VLM waypoint 训练。
>
> **最后验证**: 2026-02-25，硬件: 2× RTX PRO 6000 Blackwell (97.9 GB)，Ubuntu 24.04，CUDA 12.8
>
> **实测总耗时（clone → step=0）: ~15 分钟**（uv sync、数据下载、模型下载三路并行）

---

## Agent 行为准则

1. **`sleep` 最多 30 秒**，循环轮询长任务。
2. **后台任务用 `&`**，输出重定向文件，用 `tail` 检查。
3. **tmux send-keys 每次只发一条命令**，间隔 `sleep 2`，避免 bash 解析混乱。
4. **发现错误立刻读日志**，不要盲目重试。
5. **路径严格按照本文档**，不要自行发明。
6. **写文件前确认目录可写**（rclone 下载目录属主 `nobody:nogroup`，无写权限）。

---

## 快速 Checklist

```
□ touch ~/.no_auto_tmux
□ git config --global user.email/name
□ sudo apt-get install -y ffmpeg pkg-config build-essential libosmesa6-dev libgles2 libegl1
□ cd /workspace && git clone openpi (pytorch_lora_blackwell)
□ git submodule update --init --recursive
□ 检查 pyproject.toml av override（通常已有，无需改）

□ 【并行启动，不等待】
  □ uv sync > /tmp/uv_sync.log 2>&1 &                    (3-10 min)
  □ rclone copy libero_object_no_noops/ ... &             (~1 min)
  □ rclone copy libero_object_wp_001/ ... &               (~1 min)
  □ gsutil cp gs://openpi-assets/... pi05_base_jax/ &     (~2 min)

□ 配置 rclone gg1（下载前完成）
□ 配置 wandb → ~/.netrc
□ uv sync 完成 → 装 TF，打 transformers patch
□ 数据下载完成 → 计算 AE stats（9.2）、VLM stats（9.3）
□ gsutil 完成 → 转换 PyTorch 模型（10.3）
□ 【AE】检查路径 → 创建 tmux → 启动训练 → 验证 step=0 loss
□ 【VLM】检查路径 → 创建 tmux → 启动训练 → 验证 step=0 loss
```

**并行执行示意**：
```
clone + submodule
    ├── 【后台】uv sync ─────────────────► 装TF & patch ─────────┐
    ├── 配置 rclone gg1                                          │
    │     ├── 【后台】rclone no_noops ──► AE stats               ├──► 启动训练
    │     └── 【后台】rclone wp_001   ──► VLM stats              │
    └── 【后台】gsutil pi05_base ─────► 转换 PyTorch ────────────┘
```

---

## 1. 基础环境

```bash
touch ~/.no_auto_tmux

git config --global user.email "chuanliang.xie@gmail.com"
git config --global user.name "chuanliang"

sudo apt-get install -y ffmpeg pkg-config build-essential \
    libosmesa6-dev libgles2 libegl1
```

> 检查 `uv` 和 `rclone` 是否预装：`which uv && uv --version && rclone --version`

---

## 2. 克隆代码仓库

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
# 确认 av override 存在
grep "override-dependencies" /workspace/openpi/pyproject.toml
# 期望: "av>=13.1.0,<14.0.0"（如无见问题 1）

cd /workspace/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync > /tmp/uv_sync.log 2>&1 &
echo "PID=$!"
```

监控：`sleep 30 && tail -10 /tmp/uv_sync.log`（重复至"Installed"）

完成后：
```bash
# 安装 TF（必须 2.15.0，见问题 2）
uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"

# 打 transformers patch
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/
.venv/bin/python -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print('patch OK')
"
```

---

## 4. 配置 rclone Google Drive

Remote 名称必须为 **`gg1`**。

**本地机器**：
```bash
rclone authorize "drive"
# 浏览器授权 → 复制输出的 JSON token
```

**服务器**：
```bash
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << 'EOF'
[gg1]
type = drive
scope = drive
token = <粘贴 JSON token>
EOF

rclone lsd gg1:dissert_ntu/libero  # 验证
```

---

## 5. 下载训练数据

```bash
mkdir -p /workspace/data/libero/libero_object_no_noops \
         /workspace/data/libero/libero_object_wp_001

rclone copy gg1:dissert_ntu/libero/libero_object_no_noops/ \
    /workspace/data/libero/libero_object_no_noops/ \
    -P --transfers=8 > /tmp/rclone_rlds.log 2>&1 &

rclone copy gg1:dissert_ntu/libero/libero_object_wp_001/ \
    /workspace/data/libero/libero_object_wp_001/ \
    -P --transfers=4 > /tmp/rclone_wp.log 2>&1 &

echo "Downloads started"
```

监控：`sleep 30 && tail -3 /tmp/rclone_rlds.log && tail -3 /tmp/rclone_wp.log`

验证：
```bash
ls /workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/ | wc -l  # 34
ls /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0/ | wc -l  # 6
ls /workspace/data/libero/libero_object_wp_001/waypoint_indices.json  # 必须存在
```

---

## 6. 下载并转换 pi0.5 base 模型

### 6.1 下载 JAX checkpoint（~11.6 GB）

```bash
mkdir -p /workspace/models/pi05_base_jax
gsutil -m cp -r "gs://openpi-assets/checkpoints/pi05_base" \
    /workspace/models/pi05_base_jax/ > /tmp/gsutil.log 2>&1 &
echo "PID=$!"
```

监控：`sleep 30 && du -sh /workspace/models/pi05_base_jax/ && tail -3 /tmp/gsutil.log`

### 6.2 转换为 PyTorch

```bash
cd /workspace/openpi
.venv/bin/python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /workspace/models/pi05_base_jax/pi05_base \
    --config_name pi05_libero \
    --output_path /workspace/models/pi05_base_pytorch \
    --precision bfloat16 > /tmp/convert.log 2>&1 &
echo "PID=$!"
```

监控：`sleep 30 && tail -5 /tmp/convert.log`（完成标志：`"Model conversion completed successfully!"`）

验证：
```bash
.venv/bin/python -c "
from safetensors.torch import load_file
t = load_file('/workspace/models/pi05_base_pytorch/model.safetensors', device='cpu')
print('keys:', len(t))  # 812
"
```

---

## 7. 生成 dataset_statistics

### 7.1 AE 用 stats（从完整 RLDS）

> ⚠️ Google Drive 上的 stats 维度不匹配（15/14 维），必须从 RLDS 重算。

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
        # ⚠️ 用 'state' (8维)，不要用 'joint_state' (7维)
        all_proprios.append(step['observation']['state'].numpy().astype('float32').flatten())
all_actions = np.stack(all_actions); all_proprios = np.stack(all_proprios)
print(f'Actions: {all_actions.shape}, Proprios: {all_proprios.shape}')  # (66984,7), (66984,8)
def stats(arr):
    return {'mean': arr.mean(0).tolist(), 'std': arr.std(0).tolist(),
            'q01': np.percentile(arr,1,0).tolist(), 'q99': np.percentile(arr,99,0).tolist(),
            'min': arr.min(0).tolist(), 'max': arr.max(0).tolist()}
out = {'libero_object_no_noops': {'action': stats(all_actions), 'proprio': stats(all_proprios), 'num_samples': len(all_actions)}}
# ⚠️ 保存到可写目录（RLDS 目录属主 nobody:nogroup）
path = '/workspace/data/dataset_statistics.json'
with open(path, 'w') as f: json.dump(out, f, indent=2)
print('Saved to', path)
PYEOF
echo "AE stats PID=$!"
```

监控：`sleep 30 && tail -5 /tmp/ae_stats.log`（约 60s，完成后验证 action=7维、proprio=8维）

### 7.2 VLM 用 stats（从 waypoint-filtered RLDS）

```bash
cd /workspace/openpi
.venv/bin/python scripts/compute_wp_norm_stats.py \
    --rlds_dir /workspace/data/modified_libero_rlds/libero_object_no_noops/1.0.0 \
    --robot_type libero \
    --output_dir /workspace/data
# 约 30-40s，处理 8863 步
```

验证：
```bash
.venv/bin/python -c "
import json; d = json.load(open('/workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json'))
print('action:', len(d['action']['q99']), 'proprio:', len(d['proprio']['q99']), 'steps:', d['num_transitions'])
# 期望: 7  8  8863
"
```

---

## 8. 配置 wandb

```bash
# 新版 key (wandb_v1_...) 不能用 wandb login，用 netrc
echo "machine api.wandb.ai
  login user
  password <your_wandb_api_key>" >> ~/.netrc
chmod 600 ~/.netrc
```

> 真正验证在训练后：日志出现 `wandb: 🚀 View run at https://...` 即成功。

---

## 9. 启动训练

### 9.1 确认路径

```bash
ls /workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/dataset_info.json && echo "✓ RLDS"
ls /workspace/data/libero/libero_object_wp_001/waypoint_indices.json && echo "✓ wp_indices"
ls /workspace/data/dataset_statistics.json && echo "✓ AE stats"
ls /workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json && echo "✓ VLM stats"
ls /workspace/models/pi05_base_pytorch/model.safetensors && echo "✓ model"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### 9.2 启动 AE 训练

> **tmux 规则**: 每条 `send-keys` 只发一条命令，间隔 `sleep 2`。不要拼接 `export && torchrun`。

```bash
tmux kill-session -t waypoint_ae 2>/dev/null; sleep 1
tmux new-session -d -s waypoint_ae -x 220 -y 50

tmux send-keys -t waypoint_ae "cd /workspace/openpi" Enter; sleep 2
tmux send-keys -t waypoint_ae "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter; sleep 2
tmux send-keys -t waypoint_ae "export WANDB_API_KEY=<your_key>" Enter; sleep 2

mkdir -p /workspace/openpi/logs
tmux send-keys -t waypoint_ae ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint.py --mode ae --config configs/waypoint_ae_libero.yaml 2>&1 | tee logs/waypoint_ae_libero.log" Enter
```

### 9.3 启动 VLM 训练

| | AE | VLM |
|---|---|---|
| 模型 | PaliGemma + ActionExpert (3.6B) | PaliGemma only (2.9B) |
| Loss | MSE (flow matching) | CE (autoregressive token) |
| batch_size/GPU | 144 | **12** |
| GPU 内存 | ~55 GB | **~91-93 GB** |

```bash
tmux kill-session -t waypoint_vlm 2>/dev/null; sleep 1
tmux new-session -d -s waypoint_vlm -x 220 -y 50

tmux send-keys -t waypoint_vlm "cd /workspace/openpi" Enter; sleep 2
tmux send-keys -t waypoint_vlm "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter; sleep 2
tmux send-keys -t waypoint_vlm "export WANDB_API_KEY=<your_key>" Enter; sleep 2

tmux send-keys -t waypoint_vlm ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint.py --mode vlm --config configs/waypoint_vlm_libero.yaml 2>&1 | tee logs/waypoint_vlm_libero.log" Enter
```

> 断点续训追加 `--resume` 参数。

---

## 10. 验证训练正常

### AE 训练

```bash
sleep 30 && tail -20 /workspace/openpi/logs/waypoint_ae_libero.log
```

期望顺序出现：
1. `WaypointAEDataset: 454 episodes, 8409 valid pairs`
2. `Loaded 811 weight tensors, skipped 1`（time_mlp_in shape 变化，正常）
3. `wandb: 🚀 View run at https://...`
4. `[AE] step=0/10000 loss=0.xxx`（初始 loss 0.7-1.0，随后降至 0.2-0.3）

### VLM 训练

```bash
sleep 30 && tail -20 /workspace/openpi/logs/waypoint_vlm_libero.log
```

期望顺序出现：
1. `WaypointVLMDataset: dir=...1.0.0, M=7, stride=4, robot=libero`
2. `PaliGemma weights loaded: 603 params, 1 missing, 0 unexpected`
3. `[VLM] step=0/30000 loss=11.xxx`（初始 CE loss 11-12，正常）

速度参考：~3.1-3.3 s/step，GPU ~91-93 GB/卡。

---

## 已知问题

### 问题 1：uv sync 失败 — av 需要 ffmpeg 7

**修复**：`pyproject.toml` `[tool.uv]` 段添加：
```toml
override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74", "av>=13.1.0,<14.0.0"]
```
> `pytorch_lora_blackwell` 分支已预置，通常无需修改。

---

### 问题 2：TF 版本冲突 — `ml_dtypes has no attribute 'int2'`

**原因**：tensorflow>=2.16 与 ml_dtypes 冲突。**修复**：严格使用 `tensorflow==2.15.0`。

---

### 问题 3：action 维度广播失败 — `shapes (148,7) (15,)`

**原因**：dataset_statistics.json action 维度为 15（VLM 或其他机器人的 stats）。
**修复**：按第 7.1 节从 RLDS 重新计算。

---

### 问题 4：wandb login 报 key 长度错误

**原因**：新版 key `wandb_v1_...` 为 86 字符，旧 CLI 不支持。
**修复**：写入 `~/.netrc` 或用环境变量 `WANDB_API_KEY=<key>`。

---

### 问题 5：tmux export 解析错误 — `export: '--standalone': not a valid identifier`

**原因**：session 内有未完成的 export，后续命令被追加为 export 参数。
**修复**：`tmux kill-session -t waypoint_ae`，重建 session，每条命令加 `sleep 2`。

---

### 问题 6：VLM OOM — batch_size=16

**修复**：`configs/waypoint_vlm_libero.yaml` 设 `batch_size: 12`，并设环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

---

### 问题 7：VLM 使用系统 torchrun 报 ModuleNotFoundError

**原因**：系统 torchrun 用 python3.12，依赖在 .venv/python3.11。
**修复**：始终用 `.venv/bin/torchrun`。

---

## 关键路径速查

| 资源 | 路径 |
|------|------|
| 训练脚本 | `/workspace/openpi/scripts/train_waypoint.py` |
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
| AE Checkpoints | `/workspace/openpi/checkpoints/waypoint_ae_libero/` |
| VLM Checkpoints | `/workspace/openpi/checkpoints/waypoint_vlm_libero/` |
| Google Drive | `gg1:dissert_ntu/libero/` |

## 快速监控命令

```bash
tail -f /workspace/openpi/logs/waypoint_ae_libero.log | grep "\[AE\]"
tail -f /workspace/openpi/logs/waypoint_vlm_libero.log | grep "\[VLM\]"
watch -n 5 nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
ls -la /workspace/openpi/checkpoints/waypoint_ae_libero/
tmux attach -t waypoint_ae   # Ctrl+B, D 退出不杀进程
tmux attach -t waypoint_vlm
```
