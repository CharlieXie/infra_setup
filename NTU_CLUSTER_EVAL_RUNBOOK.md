# OpenPI v1.2_e2e — NTU Cluster LIBERO 评测 & 训练 Runbook

> **目标读者**: AI Agent。在 NTU EEE Cluster 02 上从零搭建 OpenPI eval + training 环境（无 sudo 权限）。
>
> **最后验证**: 2026-03-26，NTU Cluster (Driver 570.211.01, CUDA 12.8)，RTX PRO 6000 Blackwell
>
> **耗时预估**: 环境搭建 ~20 分钟，单 suite eval (500 ep) ~4-5 小时，training (2200 steps, 2×PRO 6000) ~TBD

---

## Agent 行为准则

1. **不要用 `sleep` 等待长命令**；用 `> logfile 2>&1 &` 后台运行，`tail -f logfile` 或轮询检查。
2. **发现错误立刻读日志**，不要盲目重试。
3. **Login 节点禁止重计算**（8GB 内存硬限制）。所有 GPU 任务必须通过 `sbatch`/`srun` 提交。
4. **断网杀进程**：srun 断网即取消；sbatch 不受影响，优先用 sbatch。
5. **不要自行安装 CUDA/GCC**，集群用 `module load`。

---

## 集群关键约束（msc 账户）

| 约束 | 值 |
|------|-----|
| 同时最多任务数 | 2 |
| 每种 GPU 最多卡数 | 2（V100 为 4） |
| srun 最长时间 | 2 小时，最多 1 GPU |
| sbatch 最长时间 | 7 天 |
| Home 目录 | `/home/chuanlia001`（50GB SSD） |
| SSD 项目 | `/projects/chuanlia001ssd`（150GB） |
| HDD 项目 | `/projects/chuanlia001`（400GB，慢，尽量不用） |
| `/tmp` | 4GB，pip install 可能撑爆，需重定向 |

---

## GPU 兼容性（PyTorch 2.7.1 cu128）

cu128 **不支持 V100 (sm_70)**，支持以下所有 GPU：

| GPU | 计算能力 | 显存 | 数量 | 适合 eval |
|-----|---------|------|------|----------|
| PRO 6000 (Blackwell) | sm_120 | 96GB | 16 | ✓ 2 GPU 并行 |
| 6000 Ada | sm_89 | 48GB | 12 | ✓ |
| A6000 | sm_86 | 48GB | 10 | ✓ |
| L40 | sm_89 | 48GB | 8 | ✓ |
| A40 | sm_86 | 48GB | 18 | ✓ |
| A5000 | sm_86 | 24GB | 8 | ✓（已验证） |
| V100 | sm_70 | 32GB | 16 | ❌ cu128 不支持 |

模型 bfloat16 占 ~12-15 GB VRAM，所有支持的 GPU（含 A5000 24GB）均可运行。

---

## 快速 Checklist — Eval

```
□ 1. git clone openpi (v1.2_e2e) + submodules
□ 2. conda create 环境 (python=3.11 + mesalib)
□ 3. 手动安装 libOSMesa.so（从 Ubuntu deb 提取，conda mesalib 不含）
□ 4. pip install PyTorch 2.7.1 cu128
□ 5. pip install JAX 0.5.3 CPU-only（仅 import 用，不参与计算）
□ 6. pip install 核心依赖 + eval 依赖 + matplotlib + LIBERO
□ 7. 打 transformers patch
□ 8. 修复 LIBERO torch.load (weights_only=False)
□ 9. 创建 eval config (更新路径)
□ 10. sbatch 提交评测
```

## 快速 Checklist — Training（在 eval 环境基础上）

```
□ 1. git clone pi_train 仓库（或确认已存在）
□ 2. pip install 训练额外依赖: torchvision, wandb, tensorflow, tensorflow-datasets
□ 3. pip install nvidia-nccl-cu12>=2.29（Blackwell 多卡 DDP 必须）
□ 4. pip install ml_dtypes>=0.4.0（修复 TF 2.15 降级导致的冲突）
□ 5. 配置 wandb（~/.netrc）
□ 6. 准备训练数据（RLDS + waypoint indices + Pi0.5 base 权重）
□ 7. 创建/更新训练 config（更新数据路径）
□ 8. srun 交互式单卡快速验证（可选）
□ 9. sbatch 提交训练（2 卡 DDP）
```

---

## 1. 克隆代码仓库

```bash
cd /projects/chuanlia001ssd/repos
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/CharlieXie/openpi.git
cd openpi
git checkout v1.2_e2e
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

验证：
```bash
git branch  # * v1.2_e2e
ls third_party/libero third_party/aloha  # 两个子模块目录存在
```

---

## 2. 创建 Conda 环境

**⚠️ 关键陷阱**: conda-forge 的 `mesalib` 包**不含** `libOSMesa.so`（只有 Vulkan driver），必须额外手动安装。仍需装 mesalib 因为它提供 GL 头文件。

```bash
# 加载 conda
module load Miniforge3 && source activate

# 重定向临时目录，防止 /tmp (4GB) 撑爆
export TMPDIR=/projects/chuanlia001ssd/tmp
export PIP_CACHE_DIR=/projects/chuanlia001ssd/tmp/pip-cache
mkdir -p $TMPDIR $PIP_CACHE_DIR

# 创建环境（装在 home SSD，~9-10GB）
conda create -p /home/chuanlia001/envs/openpi python=3.11 mesalib -c conda-forge -y
```

验证：
```bash
/home/chuanlia001/envs/openpi/bin/python --version  # Python 3.11.x
```

耗时：~5-10 分钟（网络存储较慢）。

---

## 3. 安装 libOSMesa（核心步骤）

**⚠️ 这是集群与 vast.ai 最大的差异**。集群没有 sudo，无法 `apt install libosmesa6-dev`。解决方案：直接从 Ubuntu 包提取 `.so` 文件。

```bash
mkdir -p /tmp/osmesa-extract && cd /tmp/osmesa-extract

# 下载 Ubuntu 包（不需要 sudo）
apt-get download libosmesa6
apt-get download libglapi-mesa

# 解压
dpkg-deb -x libosmesa6_*.deb extracted/
dpkg-deb -x libglapi-mesa_*.deb extracted/

# 复制到 conda env 的 lib 目录
LIBDIR=/home/chuanlia001/envs/openpi/lib
cp extracted/usr/lib/x86_64-linux-gnu/libOSMesa.so.8.0.0 "$LIBDIR/"
cp extracted/usr/lib/x86_64-linux-gnu/libglapi.so.0.0.0 "$LIBDIR/"

# 创建符号链接
cd "$LIBDIR"
ln -sf libOSMesa.so.8.0.0 libOSMesa.so.8
ln -sf libOSMesa.so.8 libOSMesa.so.6
ln -sf libOSMesa.so.8 libOSMesa.so
ln -sf libglapi.so.0.0.0 libglapi.so.0
ln -sf libglapi.so.0 libglapi.so

# 清理
rm -rf /tmp/osmesa-extract
```

验证：
```bash
ls /home/chuanlia001/envs/openpi/lib/libOSMesa.so  # 存在
```

---

## 4. 安装 Python 包

所有 pip 命令使用 conda env 内的 python：

```bash
PYTHON=/home/chuanlia001/envs/openpi/bin/python
PIP="$PYTHON -m pip"
export PIP_CACHE_DIR=/projects/chuanlia001ssd/tmp/pip-cache
export TMPDIR=/projects/chuanlia001ssd/tmp
```

### 4.1 PyTorch 2.7.1 (cu128)

```bash
$PIP install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

~2.5GB 下载，耗时 ~2-3 分钟。

### 4.2 JAX CPU-only（仅 import 用）

**背景**：openpi 的 `pi0_config.py`、`model.py`、`gemma.py` 等在模块级别 import 了 JAX/Flax，即使 eval 只用 PyTorch。装 CPU 版节省 ~5GB 空间。

```bash
$PIP install jax==0.5.3 jaxlib==0.5.3
```

### 4.3 核心依赖

```bash
$PIP install \
    "transformers==4.53.2" "flax==0.10.2" "safetensors" "sentencepiece>=0.2.0" \
    "orbax-checkpoint==0.11.13" "augmax>=0.3.4" "einops>=0.8.0" \
    "beartype==0.19.0" "jaxtyping==0.2.36" \
    "imageio>=2.36.1" "imageio-ffmpeg" "pillow>=11.0.0" "numpy>=1.22.4,<2" \
    "pyyaml" "fsspec[gcs]>=2024.6.0" "filelock>=3.16.1" "tqdm-loggable>=0.2" \
    "typing-extensions>=4.12.2" "ml_collections==1.0.0" "dm-tree>=0.1.8" \
    "pytest" "equinox>=0.11.8" "rich>=14.0.0" \
    "opencv-python>=4.10.0.84" "numpydantic>=1.6.6" "polars>=1.30.0"
```

### 4.4 评测依赖

```bash
$PIP install robosuite==1.4.1 transforms3d bddl easydict "gym==0.26.2" matplotlib
```

**⚠️ 不要漏掉 `matplotlib`**：LIBERO 的 `env_wrapper.py` 在 import 时依赖它，缺失会在运行时报 `ModuleNotFoundError`。

### 4.5 LIBERO（editable 安装）

```bash
$PIP install -e /projects/chuanlia001ssd/repos/openpi/third_party/libero
```

---

## 5. 打 Patch

### 5.1 transformers 补丁（支持 AdaRMS + KV cache 控制）

```bash
OPENPI=/projects/chuanlia001ssd/repos/openpi
cp -r $OPENPI/src/openpi/models_pytorch/transformers_replace/* \
    /home/chuanlia001/envs/openpi/lib/python3.11/site-packages/transformers/
```

验证：
```bash
$PYTHON -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print('patch OK')
"
```

### 5.2 修复 LIBERO torch.load（PyTorch 2.6+ 兼容性）

```bash
LIBERO_INIT=$OPENPI/third_party/libero/libero/libero/benchmark/__init__.py
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' "$LIBERO_INIT"
```

---

## 6. 验证环境

在 login 节点验证（不需要 GPU）：

```bash
OPENPI=/projects/chuanlia001ssd/repos/openpi
ENVDIR=/home/chuanlia001/envs/openpi
PYTHON=$ENVDIR/bin/python

MUJOCO_GL=osmesa \
LD_LIBRARY_PATH=$ENVDIR/lib:${LD_LIBRARY_PATH:-} \
PYTHONPATH=$OPENPI/src:$OPENPI/third_party/libero:${PYTHONPATH:-} \
$PYTHON -c "
print('Testing imports...')
from openpi.waypoint.normalize import NormalizationHelper, load_dataset_statistics
from openpi.waypoint.robot_config import get_robot_config
from openpi.waypoint.tokenizer import WaypointTokenizer
import openpi.models.pi0_config as pi0_config
from openpi.waypoint.joint_model import PI0WaypointJoint
from libero.libero import benchmark
bm = benchmark.get_benchmark_dict()['libero_spatial']()
print(f'LIBERO: {bm.n_tasks} tasks')
import robosuite; print('robosuite OK')
from OpenGL.GL import glGetError; print('OpenGL OK')
print('All imports OK!')
"
```

**⚠️ 注意**：`import robosuite` 在 login 节点**不设 `MUJOCO_GL=osmesa` 时会失败**（因为没有 EGL），这是正常的。必须设置 `MUJOCO_GL=osmesa` + `LD_LIBRARY_PATH`。

---

## 7. 创建评测 Config

每个 task suite 一个 YAML 文件。模板（修改 `task_suite` 和 `video_out_path`）：

```yaml
robot_type: libero
task_suite: libero_spatial  # libero_spatial | libero_object | libero_goal | libero_10

joint_checkpoint: /projects/chuanlia001ssd/models/waypoint_joint_libero_sp0.3_all_data_t1/1900
dataset_statistics_path: /projects/chuanlia001ssd/data/libero_all/dataset_statistics.json

model_action_dim: 32
model_proprio_dim: 32
horizon_steps: 32
num_waypoints: 7
vlm_max_token_len: 256
ae_max_token_len: 64

paligemma_variant: gemma_2b
action_expert_variant: gemma_300m
precision: bfloat16

norm_type: q99
torch_compile: true

num_trials_per_task: 50
num_steps_wait: 10

video_out_path: /projects/chuanlia001ssd/repos/openpi/data/libero/videos_libero_spatial
```

需创建 4 个文件：`eval_cluster_libero_spatial.yaml`、`eval_cluster_libero_object.yaml`、`eval_cluster_libero_goal.yaml`、`eval_cluster_libero_10.yaml`（仅 `task_suite` 和 `video_out_path` 不同）。

---

## 8. 提交评测

### sbatch 脚本模板

所有 sbatch 脚本**必须**包含以下环境变量：

```bash
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH="$ENVDIR/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="$OPENPI/.torch_cache"
export PYTHONPATH="$OPENPI/src:$OPENPI/third_party/libero:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
```

### 单 suite 提交

```bash
sbatch scripts/cluster_eval_full.sh libero_spatial
# 参数: libero_spatial | libero_object | libero_goal | libero_10
```

### 4 suite 全部提交（2 GPU 并行）

```bash
sbatch scripts/cluster_eval_all.sh
# 自动 2 轮并行: (libero_10 + libero_goal) → (libero_spatial + libero_object)
```

### 排除 V100

sbatch 脚本中使用以下之一：
```bash
#SBATCH --gpus=pro6000:1             # 指定 GPU 型号
#SBATCH --constraint='6000ada|a40|a5000|a6000|l40|pro6000'  # 排除 V100
```

### 监控

```bash
squeue -u $USER                                      # 查看任务状态
tail -f logs/eval_all-<JOBID>.err                    # 实时日志
grep "成功率" logs/eval_all-<JOBID>.err | tail -5    # 成功率摘要
grep "Overall" logs/eval_all-<JOBID>.err             # 最终结果
```

---

## 性能参考（实测）

| 项目 | 时间 |
|------|------|
| 模型加载（joint bfloat16） | ~50s |
| torch.compile 首次编译 | ~5-10 min（缓存后 0s） |
| 单 episode（VLM + AE 推理 + 仿真） | ~30-35s |
| 单 suite（10 tasks × 50 trials = 500 ep） | ~4-5 小时 |
| 4 suite 串行（2000 ep） | ~17-20 小时 |
| 4 suite 并行（2 GPU） | ~10 小时 |
| VRAM 占用 | ~12-15 GB |

---

## 与 vast.ai Runbook 的关键差异

| 项目 | vast.ai (V1_E2E_EVAL_RUNBOOK) | NTU Cluster (本文档) |
|------|------|------|
| 系统权限 | root / sudo | 无 sudo |
| OSMesa 安装 | `apt install libosmesa6-dev` | 从 Ubuntu deb 提取到 conda env |
| Python 环境 | uv sync (.venv) | conda + pip (~/envs/openpi) |
| JAX | 完整安装（含 CUDA） | CPU-only（仅 import 需要） |
| GPU 调度 | 直接使用 | SLURM (sbatch/srun) |
| 会话持久 | tmux | sbatch（tmux 在 login 节点断连即死） |
| CUDA 版本 | 12.8 (容器内) | 12.8 (Driver 570.211.01) |
| torch_compile 缓存 | `/workspace/openpi/.torch_cache` | `$OPENPI/.torch_cache` |
| mesalib 陷阱 | 无（apt 直接安装） | conda mesalib 不含 libOSMesa.so |
| matplotlib | 不需要额外装 | 必须额外 `pip install matplotlib` |

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `glGetError` AttributeError | 缺少 libOSMesa.so 或 LD_LIBRARY_PATH 没设 | 执行第 3 节；检查 `LD_LIBRARY_PATH` 包含 `$ENVDIR/lib` |
| `eglQueryString` AttributeError | 未设 MUJOCO_GL=osmesa，回退 EGL 但 login 节点无 GPU | 设置 `export MUJOCO_GL=osmesa` |
| `No module named 'matplotlib'` | LIBERO env_wrapper.py import 依赖 | `pip install matplotlib` |
| `QOSMaxJobsPerUserLimit` | msc 最多 2 个并发任务 | 取消一个任务后再提交 |
| `QOSMaxGRESPerUser` | 该 GPU 型号的配额用满 | 换 GPU 型号或等待 |
| CUDA available: False | 在 login 节点（无 GPU）验证 | 正常，GPU 节点上会变 True |
| torch.load error | PyTorch 2.6+ 默认 weights_only=True | 第 5.2 节 patch |
| V100 上运行崩溃 | cu128 不支持 sm_70 | 排除 V100，用 `--constraint` |

---

## 文件结构

```
/home/chuanlia001/envs/openpi/          # conda env (~9-10GB)，eval 和 training 共用
    bin/python                          # Python 3.11
    bin/torchrun                        # DDP 启动器
    lib/libOSMesa.so                    # 手动安装的 OSMesa
    lib/python3.11/site-packages/       # 所有 pip 包

/projects/chuanlia001ssd/repos/openpi/  # eval 代码仓库 (v1.2_e2e)
    configs/eval_cluster_*.yaml         # 集群专用 eval 配置
    scripts/cluster_eval_full.sh        # 单 suite sbatch 脚本
    scripts/cluster_eval_all.sh         # 4 suite sbatch 脚本
    logs/                               # sbatch 日志输出
    data/libero/videos_*/               # 评测视频输出
    .torch_cache/                       # torch.compile 缓存

/projects/chuanlia001ssd/repos/pi_train/  # training 代码仓库
    configs/waypoint_joint_libero.yaml    # 训练配置
    scripts/train_waypoint_joint.py       # Joint 训练脚本
    scripts/train_waypoint.py             # 基础训练脚本（被 joint 脚本 import）
    scripts/cluster_train_joint.sh        # sbatch 训练脚本
    logs/                                 # 训练日志
    checkpoints/                          # 训练 checkpoint 输出

/projects/chuanlia001ssd/models/          # 模型权重
    pi05_base_pytorch/model.safetensors   # Pi0.5 base 预训练权重
    waypoint_joint_libero_*/              # 已训练的 checkpoint

/projects/chuanlia001ssd/data/            # 数据
    libero_all/dataset_statistics.json    # eval 用 dataset statistics
    libero_all/whole/data/                # 训练数据根目录
        modified_libero_rlds/             # AE 用原始 RLDS
        libero_object_wp_001/             # VLM 用 waypoint 数据
            waypoint_indices.json
            waypoint_filtered_rlds__libero/
        dataset_statistics.json           # training 用 dataset statistics
```

---

---

# Part 2: Training

---

## 9. 训练环境配置（在 eval 环境基础上）

Training 和 eval 共用同一个 conda 环境 `/home/chuanlia001/envs/openpi`。eval 环境已包含大部分依赖，training 只需额外安装以下包。

### 9.1 训练代码仓库

训练代码在独立仓库 `pi_train`（非 `openpi`），需单独 clone：

```bash
cd /projects/chuanlia001ssd/repos
# git clone pi_train 仓库（具体 URL 视实际情况）
```

### 9.2 安装训练额外依赖

```bash
PYTHON=/home/chuanlia001/envs/openpi/bin/python
PIP="$PYTHON -m pip"
export TMPDIR=/projects/chuanlia001ssd/tmp
export PIP_CACHE_DIR=/projects/chuanlia001ssd/tmp/pip-cache
mkdir -p $TMPDIR $PIP_CACHE_DIR
```

#### torchvision（VLM 数据增强用）

**⚠️ 关键陷阱**：必须指定与 torch 2.7.1 匹配的版本，否则 pip 会把 torch 升级到最新版。

```bash
$PIP install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

如果 pip 意外升级了 torch，修复：
```bash
$PIP install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

#### wandb（训练日志）

```bash
$PIP install wandb
```

#### TensorFlow 2.15（读取 RLDS 数据）

```bash
$PIP install "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"
```

#### 修复 ml_dtypes 冲突

**⚠️ 关键陷阱**：TF 2.15 会把 `ml_dtypes` 降到 0.2.0，导致 JAX/tensorstore/orbax 报 `ImportError: initialization failed` 或 `cannot import name 'float8_e3m4'`。必须升级回来：

```bash
$PIP install "ml_dtypes>=0.4.0"
```

pip 会报 TF 2.15 incompatible 警告，忽略即可（TF 只用来读数据，不涉及 ml_dtypes）。

#### 升级 NCCL（Blackwell 多卡 DDP 必须）

```bash
$PIP install "nvidia-nccl-cu12>=2.29"
```

PyTorch 自带的 nccl 2.26.2 在 Blackwell (sm_120) 多卡通信时有 illegal memory access bug。pip 会报 torch incompatible 警告，忽略即可。

如果只用单卡训练，此步可跳过。

### 9.3 配置 wandb

wandb 新版 key（`wandb_v1_...`）不能用 `wandb login`，必须写 netrc：

```bash
echo "machine api.wandb.ai
  login user
  password <your_wandb_api_key>" >> ~/.netrc
chmod 600 ~/.netrc
```

验证：
```bash
grep password ~/.netrc  # 应输出 wandb API key
```

### 9.4 修复 wandb 兼容性

wandb 0.25+ 移除了 `run.mode` 属性，训练脚本中需改为 `getattr` 方式：

```python
# 旧（会报 AttributeError）:
if wandb.run and wandb.run.mode != "disabled":

# 新:
if wandb.run and getattr(wandb.run, "mode", None) != "disabled":
```

此修复已在 `pi_train/scripts/train_waypoint_joint.py` 第 194 行应用。

### 9.5 验证训练环境

```bash
PYTHON=/home/chuanlia001/envs/openpi/bin/python
$PYTHON -c "
import torch; print('torch', torch.__version__)
import torchvision; print('torchvision', torchvision.__version__)
import wandb; print('wandb OK')
import tensorflow as tf; print('TF', tf.__version__)
print('CUDA:', torch.cuda.is_available())
print('All training deps OK')
"
```

期望输出：
```
torch 2.7.1+cu128
torchvision 0.22.1+cu128
wandb OK
TF 2.15.0
CUDA: True  (在 GPU 节点上)
All training deps OK
```

---

## 10. 准备训练数据

训练需要以下数据，需从 Google Drive 或 vast.ai 传输到集群：

| 数据 | 集群路径 | 说明 |
|------|---------|------|
| 原始 RLDS (AE 用) | `/projects/chuanlia001ssd/data/libero_all/whole/data/modified_libero_rlds/libero_all_no_noops/1.0.0` | AE flow-matching 训练数据 |
| Waypoint indices | `/projects/chuanlia001ssd/data/libero_all/whole/data/libero_object_wp_001/waypoint_indices.json` | Waypoint 对索引 |
| Waypoint filtered RLDS (VLM 用) | `/projects/chuanlia001ssd/data/libero_all/whole/data/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0` | VLM 自回归训练数据 |
| Dataset statistics | `/projects/chuanlia001ssd/data/libero_all/whole/data/dataset_statistics.json` | 归一化统计量 |
| Pi0.5 base 权重 | `/projects/chuanlia001ssd/models/pi05_base_pytorch/model.safetensors` | 预训练权重 (~7.5GB) |

可用 `gdrive_sync.sh` 或 `rclone` 从 Google Drive 下载。

---

## 11. 训练配置

训练配置文件：`/projects/chuanlia001ssd/repos/pi_train/configs/waypoint_joint_libero.yaml`

关键参数：

| 参数 | 值 | 说明 |
|------|----|------|
| `original_rlds_dir` | `.../modified_libero_rlds/libero_all_no_noops/1.0.0` | AE 数据路径 |
| `wp_indices_path` | `.../libero_object_wp_001/waypoint_indices.json` | Waypoint 索引 |
| `wp_rlds_dir` | `.../waypoint_filtered_rlds__libero/1.0.0` | VLM 数据路径 |
| `dataset_statistics_path` | `.../whole/data` | 含 dataset_statistics.json 的目录 |
| `pretrained_weight_path` | `/projects/chuanlia001ssd/models/pi05_base_pytorch` | Pi0.5 base 权重目录 |
| `gradient_strategy` | `scale_gradient` | 梯度策略 |
| `gradient_scale` | `0.1` | AE 梯度缩放比例 |
| `vlm_batch_size` | `192` | 每 GPU VLM batch size |
| `ae_batch_size` | `192` | 每 GPU AE batch size |
| `num_train_steps` | `2200` | 总训练步数 |
| `peak_lr` | `9.0e-5` | 峰值学习率 |
| `save_interval` | `100` | 每 100 步保存 checkpoint |
| `wandb_project` | `waypoint_e2e` | wandb 项目名 |

---

## 12. 交互式测试训练（可选）

在正式 sbatch 提交前，可用 srun 单卡快速验证训练能启动、loss 正常：

```bash
srun --gpus pro6000:1 --time 2:00:00 --pty bash
```

进入 GPU 节点后：

```bash
cd /projects/chuanlia001ssd/repos/pi_train && PYTHONPATH=/projects/chuanlia001ssd/repos/pi_train/src:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}') /home/chuanlia001/envs/openpi/bin/python scripts/train_waypoint_joint.py --config configs/waypoint_joint_libero.yaml
```

注意：srun 最长 2 小时、最多 1 GPU，仅用于验证。正式训练用 sbatch。

---

## 13. sbatch 提交训练

### sbatch 脚本

脚本位置：`/projects/chuanlia001ssd/repos/pi_train/scripts/cluster_train_joint.sh`

```bash
#!/bin/bash
#SBATCH --job-name=joint-train
#SBATCH --gpus=pro6000:2
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/chuanlia001ssd/repos/pi_train/logs/train_%x-%j.out
#SBATCH --error=/projects/chuanlia001ssd/repos/pi_train/logs/train_%x-%j.err

CONFIG="${1:-configs/waypoint_joint_libero.yaml}"
PIDIR="/projects/chuanlia001ssd/repos/pi_train"
ENVDIR="/home/chuanlia001/envs/openpi"

cd "$PIDIR"
mkdir -p logs checkpoints

export PYTHONPATH="$PIDIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')
export PYTHONFAULTHANDLER=1

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
LOGFILE="logs/train_joint-${SLURM_JOB_ID}.log"

$ENVDIR/bin/torchrun --standalone --nnodes=1 --nproc_per_node=$NGPU \
    scripts/train_waypoint_joint.py --config "$CONFIG" 2>&1 | tee "$LOGFILE"
```

### 提交

```bash
cd /projects/chuanlia001ssd/repos/pi_train
sbatch scripts/cluster_train_joint.sh
# 或指定其他 config:
sbatch scripts/cluster_train_joint.sh configs/my_other_config.yaml
```

### 集群限制对训练的影响

| 限制 | 值 | 影响 |
|------|-----|------|
| msc 每种 GPU 最多 | 2 张 | 最多 2 卡 DDP |
| sbatch 最长时间 | 3 天 | 超长训练需 checkpoint resume |
| 同时最多任务数 | 2 | 训练 + eval 可同时跑 |
| 断网 | sbatch 不受影响 | 适合长时间训练 |

### 监控

```bash
squeue -u $USER                                                    # 查看任务状态
tail -f /projects/chuanlia001ssd/repos/pi_train/logs/train_joint-<JOBID>.log  # 实时日志
grep "\[Joint\]" logs/train_joint-<JOBID>.log | tail -5            # 最近 loss
scancel <JOBID>                                                    # 取消任务
```

### 期望初始 loss

| 指标 | 期望范围 |
|------|---------|
| VLM loss (step=0) | 11–12 |
| AE loss (step=0) | 0.7–1.0 |

### 训练日志正常启动标志

```
Training mode: joint (VLM + AE)
GPU 0: NVIDIA RTX PRO 6000 Blackwell ...
Loaded 811 weight tensors, 1 partial, skipped 0
Model: 3617.8M total, 3617.8M trainable
Syncing run waypoint_joint_libero_...
Joint Training:   0%|...
```

---

## 14. Eval 与 Training 环境差异总结

| 项目 | Eval only | Training (额外) |
|------|-----------|-----------------|
| 代码仓库 | `openpi` (v1.2_e2e) | `pi_train` |
| torchvision | 不需要 | 需要 (`0.22.1+cu128`) |
| wandb | 不需要 | 需要 |
| tensorflow | 不需要 | 需要 (`2.15.0`) |
| tensorflow-datasets | 不需要 | 需要 (`4.9.3`) |
| nvidia-nccl-cu12 | 默认即可 | 需升级 `>=2.29`（多卡） |
| ml_dtypes | 默认即可 | 需升级 `>=0.4.0`（修复 TF 冲突） |
| OSMesa / robosuite / LIBERO | 需要 | 不需要 |
| MUJOCO_GL / LD_LIBRARY_PATH | 需要 | 不需要 |
| sbatch GPU 数 | 1 | 2（DDP） |
| sbatch 命令 | `python -m openpi.waypoint.eval_libero` | `torchrun ... scripts/train_waypoint_joint.py` |

---

## 15. Training 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `No module named 'torchvision'` | eval 环境未装 torchvision | `pip install torchvision==0.22.1 --index-url .../cu128` |
| `No module named 'wandb'` | eval 环境未装 wandb | `pip install wandb` |
| `No module named 'tensorflow'` | eval 环境未装 TF | `pip install tensorflow==2.15.0 tensorflow-datasets==4.9.3` |
| `cannot import name 'float8_e3m4' from 'ml_dtypes'` | TF 2.15 降级了 ml_dtypes 到 0.2.0 | `pip install "ml_dtypes>=0.4.0"` |
| `ImportError: initialization failed` (tensorstore) | 同上，ml_dtypes 版本过低 | 同上 |
| `'Run' object has no attribute 'mode'` | wandb 0.25+ 移除了 run.mode | 改用 `getattr(wandb.run, "mode", None)` |
| pip 升级了 torch 版本 | `pip install torchvision` 未指定版本 | `pip install torch==2.7.1 torchvision==0.22.1 --index-url .../cu128` |
| pip 降级了 nccl | 安装 torchvision 时被降回 2.26.2 | 重新 `pip install "nvidia-nccl-cu12>=2.29"` |
| `QOSMaxGRESPerUser` | pro6000 配额已满（msc 最多 2 张） | 等待或换 GPU 型号 |
| TF cuDNN/cuFFT/cuBLAS 注册警告 | TF 和 PyTorch 同时加载 CUDA 库 | 无害警告，忽略 |
| TF-TRT Warning: Could not find TensorRT | 集群未装 TensorRT | 无害警告，忽略 |
