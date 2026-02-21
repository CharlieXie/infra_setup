# OpenPI Waypoint VLA — AI Agent Runbook（经验版）

> **目标读者**: AI Agent。本文档记录了在全新 vast.ai 服务器上从零到训练启动的完整操作流程，包含所有实际踩坑细节和修复方法。与 `OPENPI_WAYPOINT_VLA_SETUP.md`（设计规范版）结合使用。
>
> **最后验证**: 2026-02-21，硬件: 2× RTX PRO 6000 Blackwell (97.9 GB)，Ubuntu 24.04，CUDA 12.8
>
> **覆盖范围**: Action Expert (AE) 训练 + VLM waypoint 训练
>
> **实测总耗时（从 clone 到 step=0）: ~15 分钟**（uv sync、数据下载、模型下载三路并行）

---

## Agent 行为准则（必须遵守）

1. **`sleep` 最多 30 秒**。单次 `sleep` 不得超过 30s。需要等待长时间任务时，改用循环轮询（每次 sleep ≤ 30s，检查状态，再 sleep）。
2. **后台任务用 `&` 启动**，输出重定向到文件，之后通过 `tail` 读文件检查进度，不要用 `block_until_ms > 30000` 的阻塞式调用。
3. **tmux send-keys 每次只发一条命令**。不要在同一个 `send-keys` 里拼接多条命令，避免 bash 解析混乱。先发 `cd` 和 `export`，再单独发训练命令。
4. **发现错误立刻读日志**，不要盲目重试。每次失败后先 `tail -50 <logfile>` 定位根因。
5. **路径严格按照本文档**，不要自行发明路径。所有数据/模型/日志路径已在下方列出。

---

## 目录

1. [快速 Checklist](#1-快速-checklist)
2. [基础环境准备](#2-基础环境准备)
3. [克隆代码仓库](#3-克隆代码仓库)
4. [配置 Python 环境（uv sync）](#4-配置-python-环境uv-sync)
5. [安装额外依赖](#5-安装额外依赖)
6. [应用 transformers 补丁](#6-应用-transformers-补丁)
7. [配置 rclone Google Drive](#7-配置-rclone-google-drive)
8. [下载训练数据](#8-下载训练数据)
9. [生成 dataset_statistics（必须手动计算）](#9-生成-dataset_statistics必须手动计算)
   - 9.1–9.2: AE 用 stats（从原始 RLDS）
   - 9.3: **VLM 用 stats**（从 waypoint-filtered RLDS）
10. [下载并转换 pi0.5 base 模型](#10-下载并转换-pi05-base-模型)
11. [配置 wandb](#11-配置-wandb)
12. [启动训练](#12-启动训练)
    - 12.1–12.2: AE 训练
    - 12.4–12.5: **VLM 训练**
13. [验证训练正常](#13-验证训练正常)
    - 13.1: AE 验证
    - 13.2: **VLM 验证**
14. [已知问题与修复方案](#14-已知问题与修复方案)
    - 问题 1–6: AE / 通用
    - 问题 7–10: **VLM 专属**
15. [关键路径速查](#15-关键路径速查)

---

## 1. 快速 Checklist

```
□ touch ~/.no_auto_tmux  （重连后生效）
□ git config --global user.email/name
□ sudo apt-get install -y ffmpeg pkg-config build-essential
□ cd /workspace && git clone openpi (pytorch_lora_blackwell branch)
□ git submodule update --init --recursive
□ 检查 pyproject.toml 是否已含 av>=13.1.0,<14.0.0（通常已有，无需修改）
□ 【并行启动以下三路，不要等待】
  □ cd /workspace/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync > /tmp/uv_sync.log 2>&1 &          (3-10 min)
  □ rclone copy gg1:dissert_ntu/libero/libero_object_no_noops/ ... -P --transfers=8 &          (~1 min)
  □ rclone copy gg1:dissert_ntu/libero/libero_object_wp_001/ ... -P --transfers=4 &            (~1 min)
  □ gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base /workspace/models/pi05_base_jax/ & (~2 min)
□ 配置 rclone gg1（若还未配置，在下载前完成）
□ 配置 wandb → 写入 ~/.netrc
□ 等 uv sync 完成 → uv pip install tensorflow==2.15.0 tensorflow-datasets==4.9.3
□ cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
□ 等数据下载完成 → 手动生成 dataset_statistics.json（见第 9 节，必须做！约 50s）
□ 等 gsutil 完成 → 转换为 PyTorch: .venv/bin/python examples/convert_jax_model_to_pytorch.py ...  (~2 min)
□ 【AE 训练】检查所有路径（见第 12.1 节） → 创建 tmux session → 启动训练 → 检查 step=0 loss
□ 【VLM 训练】生成 VLM 专用 norm stats（见第 9.3 节） → 检查路径（第 12.4 节） → 启动训练（第 12.5 节） → 检查 loss
```

---

## 2. 基础环境准备

```bash
# 禁用 vast.ai 自动 tmux
touch ~/.no_auto_tmux
# 重连后生效

# 配置 git 身份
git config --global user.email "chuanliang.xie@gmail.com"
git config --global user.name "chuanliang"

# 安装构建依赖（uv sync 需要）
sudo apt-get install -y ffmpeg pkg-config build-essential
```

> **注意**: vast.ai 镜像通常预装 `uv`（`/usr/local/bin/uv`）和 `rclone`，无需重新安装。执行前先检查：
> ```bash
> which uv && uv --version
> rclone --version
> ```

---

## 3. 克隆代码仓库

```bash
cd /workspace

# openpi 主仓库（含 waypoint VLA 实现）
git clone https://<PAT>@github.com/CharlieXie/openpi.git
cd openpi
git checkout pytorch_lora_blackwell
git submodule update --init --recursive

# infra_setup（本文档所在仓库）
cd /workspace
git clone https://<PAT>@github.com/CharlieXie/infra_setup.git
```

验证分支：
```bash
cd /workspace/openpi && git branch  # 应显示 * pytorch_lora_blackwell
```

---

## 4. 配置 Python 环境（uv sync）

### 4.1 确认 `av` 版本 override 是否已存在

`openpi` 通过 `lerobot` 依赖 `av` 包。`av >= 14.0` 要求 ffmpeg 7 从源码编译，而 Ubuntu 22/24 系统只有 ffmpeg 6。需要在 `pyproject.toml` 的 `[tool.uv]` 中添加 override，强制使用有预编译 wheel 的 `av 13.x`。

**先检查是否已存在（`pytorch_lora_blackwell` 分支通常已预置，无需修改）：**
```bash
grep "override-dependencies" /workspace/openpi/pyproject.toml
# 期望输出包含: "av>=13.1.0,<14.0.0"
```

如果输出中**已包含** `av>=13.1.0,<14.0.0`，跳过下面的修复步骤，直接进入 4.2。

如果**不包含**，执行修复：
```bash
sed -i 's/override-dependencies = \["ml-dtypes==0.4.1", "tensorstore==0.1.74"\]/override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74", "av>=13.1.0,<14.0.0"]/' /workspace/openpi/pyproject.toml
# 验证修改
grep "override-dependencies" /workspace/openpi/pyproject.toml
```

### 4.2 后台运行 uv sync

```bash
cd /workspace/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync > /tmp/uv_sync.log 2>&1 &
echo "uv sync PID=$!"
```

监控（每次 sleep ≤ 30s）：
```bash
# 每隔 30s 检查一次
sleep 30 && tail -10 /tmp/uv_sync.log
# 重复直到看到 "Resolved" / "Installed" 或 error
```

> **实测耗时**: vast.ai 服务器上约 **3–10 分钟**（取决于网速和缓存状态）。文档旧版预估 10–20 分钟偏保守。**建议 uv sync 后台运行的同时立即启动数据下载和模型下载（见并行执行建议）。**

完成后验证：
```bash
/workspace/openpi/.venv/bin/python -c "import torch; print(torch.__version__)"   # 2.7.x+cu128
/workspace/openpi/.venv/bin/python -c "import jax; print(jax.__version__)"       # 0.5.x
/workspace/openpi/.venv/bin/python -c "import transformers; print(transformers.__version__)"  # 4.53.x
```

---

## ⚡ 并行执行建议（节省 ~30 分钟）

各步骤之间存在依赖关系，但多个耗时操作可以并行。**推荐执行顺序**：

```
克隆 openpi & submodule
        │
        ├──► 【后台】uv sync ──────────────────────────► 装TF & 打patch ──────┐
        │                                                                     │
        ├──► 配置 rclone gg1                                                  │
        │        │                                                            ▼
        │        ├──► 【后台】rclone 下载 libero_object_no_noops ──► 计算stats  ├──► 启动训练
        │        └──► 【后台】rclone 下载 libero_object_wp_001                 │
        │                                                                     │
        └──► 【后台】gsutil 下载 JAX checkpoint ──► 转换 PyTorch ─────────────┘
```

- `uv sync`、`rclone` 下载、`gsutil` 下载三路**同时在后台启动**，实测总耗时约 15 分钟
- `uv sync` 完成后立即执行第 5、6 节（装 TF、打 patch）
- `rclone` 下载完成后立即执行第 9 节（计算 stats）
- `gsutil` 下载完成后立即执行第 10.3 节（模型转换）

---

## 5. 安装额外依赖

`uv sync` 不包含 TensorFlow，但 RLDS 数据加载必须用它：

```bash
cd /workspace/openpi
uv pip install --python .venv/bin/python \
    "tensorflow==2.15.0" \
    "tensorflow-datasets==4.9.3"
```

> **⚠️ 必须用 2.15.0**，不能用更新版本。`tensorflow>=2.16` 与 `ml_dtypes` 有冲突，会报：
> `AttributeError: module 'ml_dtypes' has no attribute 'int2'`

验证：
```bash
/workspace/openpi/.venv/bin/python -c "
import tensorflow as tf; import tensorflow_datasets as tfds
print('TF:', tf.__version__, 'TFDS:', tfds.__version__)
"
# 期望: TF: 2.15.0 TFDS: 4.9.3
```

---

## 6. 应用 transformers 补丁

```bash
cd /workspace/openpi
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/
```

验证：
```bash
.venv/bin/python -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print('transformers patch OK')
"
```

---

## 7. 配置 rclone Google Drive

Remote 名称必须为 **`gg1`**（训练配置文件和本文档所有命令都用这个名字）。

### Headless 服务器 OAuth 流程

在**本地机器**运行：
```bash
rclone authorize "drive"
# 浏览器授权后，终端输出 JSON token，复制整段
```

在**服务器**写入配置：
```bash
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << 'EOF'
[gg1]
type = drive
scope = drive
token = <粘贴上面的 JSON token>
EOF
```

验证：
```bash
rclone lsd gg1:  # 应该列出 Google Drive 根目录
rclone lsd gg1:dissert_ntu/libero  # 应该看到 libero_object_no_noops 和 libero_object_wp_001
```

---

## 8. 下载训练数据

```bash
mkdir -p /workspace/data/libero/libero_object_no_noops \
         /workspace/data/libero/libero_object_wp_001

# 后台下载，两个任务并行
rclone copy gg1:dissert_ntu/libero/libero_object_no_noops/ \
    /workspace/data/libero/libero_object_no_noops/ \
    -P --transfers=8 > /tmp/rclone_rlds.log 2>&1 &

rclone copy gg1:dissert_ntu/libero/libero_object_wp_001/ \
    /workspace/data/libero/libero_object_wp_001/ \
    -P --transfers=4 > /tmp/rclone_wp.log 2>&1 &

echo "Both downloads started"
```

监控进度（≤ 30s sleep）：
```bash
sleep 30 && tail -5 /tmp/rclone_rlds.log && tail -5 /tmp/rclone_wp.log
```

完成后验证文件数量：
```bash
ls /workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/ | wc -l
# 期望: 34  (32 个 tfrecord + dataset_info.json + features.json)

ls /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0/ | wc -l
# 期望: 6  (4 个 tfrecord + dataset_info.json + features.json)

ls /workspace/data/libero/libero_object_wp_001/waypoint_indices.json
# 必须存在
```

---

## 9. 生成 dataset_statistics（必须手动计算）

> **⚠️ 关键陷阱！** Google Drive 上存储的 `dataset_statistics.json` 文件有多个版本，大多数是为**其他任务**（VLM 训练/R1 机器人）生成的，action 维度不匹配：
>
> | 文件来源 | action 维度 | 用途 |
> |---------|------------|------|
> | `models/traind_vlm_models/dataset_statistics.json` | 15 维 | VLM 训练（waypoint filtered） |
> | `models/trained_action_expert_1/dataset_statistics.json` | 14 维 | R1 Lite 机器人 |
> | `models/libero_ar/dataset_statistics.json` | 9 维 | 其他实验 |
>
> **AE 训练需要的是 LIBERO 原始 action = 7 维，proprio = 8 维。** 必须从下载的 RLDS 数据重新计算。

### 9.1 先确认 LIBERO 的 observation key

LIBERO RLDS 中：
- action key: `"action"` (7 维: 6 关节 + 1 夹爪)
- proprio key: **`"state"`** (8 维: 7 关节位置 + 1 夹爪位置)

> **⚠️ 不要用 `"joint_state"` key**，那只有 7 维（缺少夹爪）。`robot_config.py` 里 `make_libero_config()` 指定的是 `state_obs_keys=["state"]`，`actual_proprio_dim=8`。

### 9.2 运行计算脚本

```bash
cd /workspace/openpi
mkdir -p /workspace/data/libero_object_no_noops/1.0.0

.venv/bin/python - << 'PYEOF'
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import numpy as np, json

print('Loading LIBERO RLDS...')
b = tfds.builder_from_directory(
    '/workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0')
ds = b.as_dataset(split='train')

all_actions, all_proprios = [], []
for ep in ds:
    for step in ep['steps']:
        all_actions.append(step['action'].numpy().astype('float32'))
        all_proprios.append(step['observation']['state'].numpy().astype('float32').flatten())

all_actions = np.stack(all_actions)
all_proprios = np.stack(all_proprios)
print(f'Actions: {all_actions.shape}, Proprios: {all_proprios.shape}')
# 期望: Actions: (66984, 7), Proprios: (66984, 8)

def stats(arr):
    return {
        'mean': arr.mean(0).tolist(), 'std': arr.std(0).tolist(),
        'q01': np.percentile(arr, 1, 0).tolist(),
        'q99': np.percentile(arr, 99, 0).tolist(),
        'min': arr.min(0).tolist(), 'max': arr.max(0).tolist(),
    }

out = {'libero_object_no_noops': {
    'action': stats(all_actions),
    'proprio': stats(all_proprios),
    'num_samples': len(all_actions),
}}
path = '/workspace/data/libero_object_no_noops/1.0.0/dataset_statistics.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print('Saved to', path)
print('action q99:', out['libero_object_no_noops']['action']['q99'])
print('proprio q99 dims:', len(out['libero_object_no_noops']['proprio']['q99']))
# 期望: action q99 = 7 values, proprio q99 = 8 values
PYEOF
```

> 这个脚本约需 **45–60 秒**，处理 66984 个时间步。

验证：
```bash
python3 -c "
import json
d = json.load(open('/workspace/data/libero_object_no_noops/1.0.0/dataset_statistics.json'))
k = list(d.keys())[0]
print('dataset:', k)
print('action q99 dims:', len(d[k]['action']['q99']))   # 必须是 7
print('proprio q99 dims:', len(d[k]['proprio']['q99'])) # 必须是 8
"
```

### 9.3 生成 VLM 专用归一化统计量

> **VLM 训练使用的 stats 与 AE 不同。** VLM 从 waypoint-filtered RLDS 计算统计量，保存到独立路径。

```bash
cd /workspace/openpi
.venv/bin/python scripts/compute_wp_norm_stats.py \
    --rlds_dir /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0 \
    --robot_type libero \
    --output_dir /workspace/data/libero/libero_object_wp_001/norm_stats
```

> 脚本约需 **30–40 秒**，处理 454 episodes, 8863 步。

验证：
```bash
python3 -c "
import json
d = json.load(open('/workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json'))
print('action q99 dims:', len(d['action']['q99']))   # 必须是 7
print('proprio q99 dims:', len(d['proprio']['q99'])) # 必须是 8
print('num_transitions:', d['num_transitions'])       # 8863
"
```

> **如果 `scripts/compute_wp_norm_stats.py` 不存在**，可以用 inline 方式生成：
> ```bash
> cd /workspace/openpi
> mkdir -p /workspace/data/libero/libero_object_wp_001/norm_stats
> .venv/bin/python - << 'PYEOF'
> import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
> import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
> import tensorflow_datasets as tfds
> import numpy as np, json
> b = tfds.builder_from_directory(
>     '/workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0')
> ds = b.as_dataset(split='train')
> all_a, all_p = [], []
> for ep in ds:
>     for s in ep['steps']:
>         all_a.append(s['action'].numpy().astype('float32'))
>         all_p.append(s['observation']['state'].numpy().astype('float32').flatten())
> all_a, all_p = np.stack(all_a), np.stack(all_p)
> def st(arr):
>     return {'mean':arr.mean(0).tolist(),'std':arr.std(0).tolist(),
>             'q01':np.percentile(arr,1,0).tolist(),'q99':np.percentile(arr,99,0).tolist(),
>             'min':arr.min(0).tolist(),'max':arr.max(0).tolist()}
> out = {'action':st(all_a),'proprio':st(all_p),'num_transitions':len(all_a),'num_trajectories':454}
> path = '/workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json'
> with open(path,'w') as f: json.dump(out,f,indent=2)
> print(f'Saved to {path}, {len(all_a)} steps')
> PYEOF
> ```

---

## 10. 下载并转换 pi0.5 base 模型

> **注意**: Google Drive `dissert_ntu/models/` 中**没有** pi05 base PyTorch 模型。必须从 GCS 公开 bucket 下载 JAX checkpoint 并转换。

### 10.1 安装 gsutil

```bash
pip3 install gsutil  # 如果没有
```

### 10.2 后台下载 JAX checkpoint (~11.6 GB)

```bash
mkdir -p /workspace/models/pi05_base_jax
gsutil -m cp -r "gs://openpi-assets/checkpoints/pi05_base" \
    /workspace/models/pi05_base_jax/ > /tmp/gsutil.log 2>&1 &
echo "Download PID=$!"
```

监控：
```bash
sleep 30 && du -sh /workspace/models/pi05_base_jax/ && tail -3 /tmp/gsutil.log
```

> **实测耗时**: vast.ai 服务器下载 GCS 公开 bucket 速度约 **200–500 MiB/s**，11.6 GB 约需 **1–3 分钟**。建议和 `uv sync`、`rclone` 下载同时后台启动。

下载完成后约 12 GB，验证：
```bash
ls /workspace/models/pi05_base_jax/pi05_base/params/
# 应该有 ocdbt.process_0/ 目录和 commit_success.txt
```

### 10.3 转换为 PyTorch 格式

```bash
cd /workspace/openpi
.venv/bin/python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /workspace/models/pi05_base_jax/pi05_base \
    --config_name pi05_libero \
    --output_path /workspace/models/pi05_base_pytorch \
    --precision bfloat16 > /tmp/convert.log 2>&1 &
echo "Conversion PID=$!"
```

> **config_name 必须用 `pi05_libero`**，不能用 `pi05_base`（不存在该 config）。转换约需 **1.5–2 分钟**。

监控：
```bash
sleep 30 && tail -5 /tmp/convert.log
# 成功标志: "Model conversion completed successfully!"
```

验证输出：
```bash
cd /workspace/openpi && .venv/bin/python -c "
from safetensors.torch import load_file
t = load_file('/workspace/models/pi05_base_pytorch/model.safetensors', device='cpu')
print(f'Total keys: {len(t)}')           # 812
print('action_in_proj:', t['action_in_proj.weight'].shape)  # [1024, 32]
print('time_mlp_in:', t['time_mlp_in.weight'].shape)        # [1024, 1024]
"
```

---

## 11. 配置 wandb

wandb 使用新版 API key（`wandb_v1_` 前缀），**不能用 `wandb login` 命令**（只接受 40 字符旧格式）。改用环境变量或写入 netrc：

```bash
# 方法 A：环境变量（推荐，每次启动训练时设置）
export WANDB_API_KEY=<your_wandb_api_key>

# 方法 B：写入 netrc（持久化）
echo "machine api.wandb.ai
  login user
  password <your_wandb_api_key>" >> ~/.netrc
chmod 600 ~/.netrc
```

验证：

> **注意**: `wandb.Api().viewer` 在新版 key 下可能返回异常，**不建议**用此命令做预验证。真正的验证在训练启动后——日志中出现 `wandb: 🚀 View run at https://...` 即表示连接成功（见第 13 节）。
>
> 如需提前确认 key 有效，检查 `~/.netrc` 文件中的内容是否正确写入即可：
> ```bash
> grep -A2 "api.wandb.ai" ~/.netrc
> ```

---

## 12. 启动训练

### 12.1 确认所有路径存在

```bash
# 一次性检查所有必需路径
ls /workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/dataset_info.json && echo "✓ RLDS"
ls /workspace/data/libero/libero_object_wp_001/waypoint_indices.json && echo "✓ waypoint_indices"
ls /workspace/data/libero_object_no_noops/1.0.0/dataset_statistics.json && echo "✓ stats"
ls /workspace/models/pi05_base_pytorch/model.safetensors && echo "✓ model"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### 12.2 创建 tmux session 并启动

> **⚠️ tmux 操作规则**:
> - 每次 `send-keys` **只发一条命令**，用 `sleep 2` 间隔等 bash 执行完
> - 不要拼接 `export VAR=... && torchrun ...`，会导致 bash 解析错误
> - 如果 session 之前崩溃过，先 `tmux kill-session -t waypoint_ae` 再重建

```bash
# 创建新 session
tmux kill-session -t waypoint_ae 2>/dev/null; sleep 1
tmux new-session -d -s waypoint_ae -x 220 -y 50

# 每条 send-keys 只发一个命令，之间 sleep 2 等 bash 执行完
tmux send-keys -t waypoint_ae "cd /workspace/openpi" Enter
sleep 2
tmux send-keys -t waypoint_ae "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter
sleep 2
tmux send-keys -t waypoint_ae "export WANDB_API_KEY=<your_key>" Enter
sleep 2

# 创建日志目录（在 tmux 外执行即可）
mkdir -p /workspace/openpi/logs

# 启动训练
tmux send-keys -t waypoint_ae ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint.py --mode ae --config configs/waypoint_ae_libero.yaml 2>&1 | tee logs/waypoint_ae_libero.log" Enter
```

### 12.4 确认 VLM 训练路径

```bash
ls /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0/dataset_info.json && echo "✓ VLM RLDS"
ls /workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json && echo "✓ VLM stats"
ls /workspace/models/pi05_base_pytorch/model.safetensors && echo "✓ model"
```

### 12.5 启动 VLM 训练

> **VLM 与 AE 训练的关键区别**:
>
> | | AE | VLM |
> |---|---|---|
> | 模型 | PaliGemma + ActionExpert (3.6B) | PaliGemma only (2.9B) |
> | Loss | MSE (flow matching) | CE (autoregressive token) |
> | batch_size (per GPU) | 144 | **12**（VLM 序列更长, 全量 finetune 2.9B 需更多内存） |
> | GPU 内存 | ~50-60 GB | **~91-93 GB**（必须设 `expandable_segments`） |
> | Gradient Checkpointing | 手动逐层 checkpoint (gemma_pytorch.py) | HuggingFace API 自动 checkpoint |
> | 数据启动 | ~10s（early-yield） | ~8s（同样 early-yield） |
>
> **必须**使用 `.venv/bin/torchrun`（系统 torchrun 使用错误的 Python 解释器）。
> **必须**设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（否则 CUDA 内存碎片化导致 OOM）。

```bash
# 创建新 session（如果已有同名 session，先杀掉）
tmux kill-session -t waypoint_vlm 2>/dev/null; sleep 1
tmux new-session -d -s waypoint_vlm -x 220 -y 50

# 逐条发送命令
tmux send-keys -t waypoint_vlm "cd /workspace/openpi" Enter
sleep 2
tmux send-keys -t waypoint_vlm "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter
sleep 2
tmux send-keys -t waypoint_vlm "export WANDB_API_KEY=<your_key>" Enter
sleep 2

# 创建日志目录
mkdir -p /workspace/openpi/logs

# 启动 VLM 训练
tmux send-keys -t waypoint_vlm ".venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint.py --mode vlm --config configs/waypoint_vlm_libero.yaml 2>&1 | tee logs/waypoint_vlm_libero.log" Enter
```

> **断点续训**: 追加 `--resume` 参数即可从最新 checkpoint 恢复。

---

## 13. 验证训练正常

### 13.1 验证 AE 训练

等待 30 秒后开始检查：

```bash
sleep 30 && tail -20 /workspace/openpi/logs/waypoint_ae_libero.log
```

按顺序应出现以下关键行：

| 顺序 | 关键日志 | 含义 |
|-----|---------|------|
| 1 | `WaypointAEDataset: 454 episodes, 8409 valid pairs` | 数据集加载成功 |
| 2 | `Loaded 811 weight tensors, skipped 1` | pi0.5 权重加载（time_mlp_in 因 shape 变化被跳过，正常） |
| 3 | `Constructing tf.data.Dataset libero_object` | RLDS 数据读取开始 |
| 4 | `wandb: 🚀 View run at https://...` | wandb 连接成功 |
| 5 | `Model: 3617.8M total, 3617.8M trainable` | 模型初始化成功 |
| 6 | `[AE] step=0/10000 loss=0.xxx` | **第一步 loss，训练开始** |

初始 loss 应在 0.7–1.0 范围，随后快速下降到 0.2–0.3。

如果 30 秒后日志还在 step=0，说明 RLDS 数据管道在初始化，继续等待（最多 90 秒）。

### 13.2 验证 VLM 训练

```bash
sleep 30 && tail -20 /workspace/openpi/logs/waypoint_vlm_libero.log
```

按顺序应出现以下关键行：

| 顺序 | 关键日志 | 含义 |
|-----|---------|------|
| 1 | `WaypointVLMDataset: dir=...1.0.0, M=7, stride=4, robot=libero` | 数据集创建成功 |
| 2 | `PaliGemma weights loaded: 603 params, 1 missing, 0 unexpected` | 权重加载成功（1 missing 正常） |
| 3 | `Model: 2923.3M total, 2923.3M trainable` | 模型初始化成功 |
| 4 | `wandb: 🚀 View run at https://...` | wandb 连接成功 |
| 5 | `Constructing tf.data.Dataset waypoint_filtered_rlds` | RLDS 数据读取 |
| 6 | `[VLM] step=0/30000 loss=11.xxx` | **第一步 loss，训练开始** |

**VLM 关键指标**:
- 初始 CE loss 应在 **11–12** 范围（正常，因为 vocab size 很大）
- 前 50 步快速下降到 **6–7**，500 步后到 **4–5**
- 速度约 **3.1–3.3 s/step**（DDP 2 卡，batch_size=12/GPU）
- GPU 内存约 **91–93 GB** per GPU（正常，非常接近上限）

```bash
# 检查 GPU 使用
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
# 期望: 两卡各 ~91000-93000 MiB / 97887 MiB
```

> **如果 VLM 启动后几秒就 OOM**，检查：
> 1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 是否设置
> 2. 是否使用了 `.venv/bin/torchrun`（不是系统 torchrun）
> 3. batch_size 是否为 12（config 文件中确认）

---

## 14. 已知问题与修复方案

### 问题 1：`uv sync` 失败 — `av` 包需要 ffmpeg 7

```
Warning! You are installing from source. It is EXPECTED that it will fail. 
You are REQUIRED to use ffmpeg 7.
```

**原因**: `lerobot` 依赖 `av>=14`，而 Ubuntu 22/24 系统自带 ffmpeg 6，无预编译 wheel。

**修复**: 在 `pyproject.toml` `[tool.uv]` 段添加 override：
```toml
override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74", "av>=13.1.0,<14.0.0"]
```
`av 13.1.0` 有预编译 manylinux wheel，兼容 ffmpeg 6。

> **注意**: `pytorch_lora_blackwell` 分支已预置此 override，通常无需手动修改。遇到此报错前先用 `grep "override-dependencies" pyproject.toml` 确认（见第 4.1 节）。

---

### 问题 2：训练崩溃 — action 维度广播失败

```
ValueError: operands could not be broadcast together with shapes (148,7) (15,)
```

**原因**: `dataset_statistics.json` 里 action 维度（15）与 LIBERO 实际 action（7）不匹配。Google Drive 上存了多份 stats 文件，大多数是为 VLM 训练或其他机器人生成的，不能直接用于 LIBERO AE 训练。

**修复**: 从 LIBERO RLDS 数据重新计算 stats（见第 9 节）。

---

### 问题 3：proprio 维度错误（7 维而非 8 维）

**原因**: LIBERO RLDS observation 中有 `"joint_state"`（7 维）和 `"state"`（8 维）两个 key，容易搞混。`robot_config.py` 指定用 `"state"`，包含 7 关节 + 1 夹爪 = 8 维。

**修复**: 计算 stats 时明确用 `step['observation']['state']`，不用 `joint_state`。

---

### 问题 4：`wandb login` 报 key 长度错误

```
ValueError: API key must be 40 characters long, yours was 86
```

**原因**: 新版 wandb API key 格式为 `wandb_v1_...`（86 字符），旧版 CLI 不支持。

**修复**: 用环境变量 `WANDB_API_KEY=<key>` 代替 `wandb login`，或写入 `~/.netrc`。

---

### 问题 5：tmux 训练命令被 bash 解析为 `export` 参数

```
-bash: export: `--standalone': not a valid identifier
```

**原因**: 在之前的 tmux session 中有未完成的 `export` 命令，后续 `send-keys` 发的训练命令被追加到了 `export` 语句后面。

**修复**: 发现异常时，先 `tmux kill-session -t waypoint_ae` 杀掉旧 session，再重建，每条 `send-keys` 间加 `sleep 2`。

---

### 问题 6：训练速度 ~14s/step（AE，正常无需担心）

当前硬件（2× RTX PRO 6000 Blackwell），batch_size=144，全量 finetune 3.6B 参数时：
- 前几步较慢（RLDS 数据 prefetch 未 warm up）
- 稳定后约 13–14 s/step
- 预计总训练时间（10000 steps）约 **40–45 小时**

---

### 问题 7：VLM 训练 OOM — batch_size=16 DDP 报 CUDA out of memory

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.68 GiB.
```

**原因**: VLM 全量 finetune PaliGemma 2B 时，模型权重 + 优化器状态 + 激活 ≈ 91 GB/GPU。DDP 增加梯度同步缓冲和跨 rank CUDA context，batch_size=16 超出单卡 ~95 GB 的容量。

**修复**: 将 batch_size 设为 **12**（`configs/waypoint_vlm_libero.yaml`），DDP 2 卡有效 batch=24。同时必须设置环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

---

### 问题 8：VLM 使用系统 torchrun 启动报 ModuleNotFoundError

```
ModuleNotFoundError: No module named 'safetensors'
```

**原因**: 系统 `torchrun`（`/venv/main/bin/torchrun`）使用系统 Python (`python3.12`)，而项目依赖安装在 `.venv` (python3.11) 中。

**修复**: 始终使用 `.venv/bin/torchrun` 启动 VLM DDP 训练。AE 训练同理。

---

### 问题 9：VLM 启动极慢（>5 分钟才出 step=0）

**原因**: `vlm_dataset.py` 的 shuffle buffer 必须完全填满才开始 yield 第一个 batch。5000 条 buffer 需要遍历 RLDS 约 5 轮。

**修复**: `vlm_dataset.py` 的 `__iter__` 方法应使用 early-yield 策略（buffer 有 32 条即开始 yield），与 `ae_dataset.py` 一致。此修复已合入 `pytorch_lora_blackwell` 分支。若遇到此问题，检查 `vlm_dataset.py` 第 190 行附近的 `__iter__` 是否有 `min(32, self.shuffle_buffer_size)` 逻辑。

---

### 问题 10：VLM Gradient Checkpointing 无效 — batch_size=4 就 OOM

**原因**: `vlm_model.py` 中 `gradient_checkpointing_enable()` 仅手动设置 `self.paligemma.language_model.gradient_checkpointing = True`，这只会禁用 KV cache，**不会**减少激活内存。HuggingFace 的 `GemmaDecoderLayer` 继承自 `GradientCheckpointingLayer`，需要通过 `model.gradient_checkpointing_enable()` API 激活才能真正在每层 `__call__` 中使用 checkpoint。

**修复**: `vlm_model.py` 的 `gradient_checkpointing_enable()` 方法应调用：
```python
self.paligemma.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```
此修复已合入 `pytorch_lora_blackwell` 分支。修复后 batch_size=16 单卡可用，batch_size=12 DDP 可用。

---

## 15. 关键路径速查

| 资源 | 路径 |
|------|------|
| openpi 代码 | `/workspace/openpi/` |
| 训练脚本 | `/workspace/openpi/scripts/train_waypoint.py` |
| AE 训练配置 | `/workspace/openpi/configs/waypoint_ae_libero.yaml` |
| VLM 训练配置 | `/workspace/openpi/configs/waypoint_vlm_libero.yaml` |
| Pi0.5 PyTorch 权重 | `/workspace/models/pi05_base_pytorch/model.safetensors` |
| Pi0.5 JAX 原始 checkpoint | `/workspace/models/pi05_base_jax/pi05_base/` |
| LIBERO RLDS 原始数据 | `/workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/` |
| Waypoint indices | `/workspace/data/libero/libero_object_wp_001/waypoint_indices.json` |
| Waypoint filtered RLDS（VLM 用） | `/workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0/` |
| **Dataset statistics（AE 用）** | `/workspace/data/libero_object_no_noops/1.0.0/dataset_statistics.json` |
| **Dataset statistics（VLM 用）** | `/workspace/data/libero/libero_object_wp_001/norm_stats/dataset_statistics.json` |
| AE 训练日志 | `/workspace/openpi/logs/waypoint_ae_libero.log` |
| VLM 训练日志 | `/workspace/openpi/logs/waypoint_vlm_libero.log` |
| AE Checkpoints | `/workspace/openpi/checkpoints/waypoint_ae_libero/` |
| VLM Checkpoints | `/workspace/openpi/checkpoints/waypoint_vlm_libero/` |
| Google Drive 数据源 | `gg1:dissert_ntu/libero/` |
| Google Drive 模型存档 | `gg1:dissert_ntu/models/` |

---

## 附录：快速监控命令

```bash
# 实时 AE 训练进度
tail -f /workspace/openpi/logs/waypoint_ae_libero.log | grep "\[AE\]"

# 实时 VLM 训练进度
tail -f /workspace/openpi/logs/waypoint_vlm_libero.log | grep "\[VLM\]"

# GPU 状态
watch -n 5 nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader

# 检查 checkpoint 是否保存
ls -la /workspace/openpi/checkpoints/waypoint_ae_libero/
ls -la /workspace/openpi/checkpoints/waypoint_vlm_libero/

# 查看 tmux session
tmux attach -t waypoint_ae   # AE 训练
tmux attach -t waypoint_vlm  # VLM 训练
# 退出不杀进程: Ctrl+B, D
```
