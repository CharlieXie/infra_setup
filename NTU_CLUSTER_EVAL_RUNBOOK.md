# NTU Cluster LIBERO Eval Runbook

> AI Agent 参考文档。读完即可在 NTU EEE Cluster 02 上提交 LIBERO 评测。
>
> 集群完整手册：`/projects/chuanlia001ssd/docs/ai-cluster-manual.md`
> 训练 Runbook：`NTU_CLUSTER_TRAIN_RUNBOOK.md`

---

## 集群关键约束（msc 账户）

| 约束 | 值 |
|------|-----|
| 用户名 | `chuanlia001` |
| 同时最多任务数 | 2 |
| 每种 GPU 最多 | 2 张（V100 为 4） |
| sbatch 最长 | **3 天**（不指定默认 1 小时） |
| srun 最长 | 2 小时，最多 1 GPU（仅调试用） |
| 推荐 GPU | 除 V100 外均可（cu128 不支持 sm_70） |
| Login 节点 | 8GB 内存限制，禁止重计算，断连杀进程 |

---

## 文件结构

```
/home/chuanlia001/envs/openpi/             # conda 环境（eval + training 共用）
    lib/libOSMesa.so                       # 手动安装的 OSMesa（关键）

/projects/chuanlia001ssd/repos/openpi/     # eval 代码 (v1.2_e2e)
    configs/eval_cluster_*.yaml            # 集群专用 eval 配置
    scripts/cluster_eval_full.sh           # 单 suite sbatch 脚本
    scripts/cluster_eval_all.sh            # 4 suite sbatch 脚本
    logs/                                  # sbatch 日志
    data/libero/videos_*/                  # 评测视频输出
    .torch_cache/                          # torch.compile 缓存

/projects/chuanlia001ssd/models/           # checkpoint 权重
/projects/chuanlia001ssd/data/             # eval/training 数据
    libero_all/dataset_statistics.json     # eval 用 dataset statistics
```

---

## 提交评测

### 单 suite

```bash
cd /projects/chuanlia001ssd/repos/openpi
sbatch scripts/cluster_eval_full.sh libero_spatial
# 可选: libero_spatial | libero_object | libero_goal | libero_10
```

### 4 suite 全部（2 GPU 并行）

```bash
sbatch scripts/cluster_eval_all.sh
# 自动 2 轮并行: (libero_10 + libero_goal) → (libero_spatial + libero_object)
```

### 排除 V100

sbatch 脚本中使用：
```bash
#SBATCH --gpus=pro6000:1                                       # 指定型号
#SBATCH --constraint='6000ada|a40|a5000|a6000|l40|pro6000'     # 或排除 V100
```

---

## 监控

```bash
squeue -u chuanlia001                                  # 任务状态
tail -f logs/eval_all-<JOBID>.err                      # 实时日志
grep "成功率" logs/eval_all-<JOBID>.err | tail -5      # 成功率
grep "Overall" logs/eval_all-<JOBID>.err               # 最终结果
scancel <JOBID>                                        # 取消
```

---

## Eval Config 模板

每个 task suite 一个 YAML，仅 `task_suite` 和 `video_out_path` 不同：

```yaml
robot_type: libero
task_suite: libero_spatial    # libero_spatial | libero_object | libero_goal | libero_10
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

---

## 性能参考

| 项目 | 时间 |
|------|------|
| 模型加载 | ~50s |
| torch.compile 首次编译 | ~5-10 min（缓存后 0s） |
| 单 episode | ~30-35s |
| 单 suite（500 ep） | ~4-5 小时 |
| 4 suite 并行（2 GPU） | ~10 小时 |
| VRAM 占用 | ~12-15 GB |

---

## sbatch 环境变量（脚本必须包含）

```bash
OPENPI="/projects/chuanlia001ssd/repos/openpi"
ENVDIR="/home/chuanlia001/envs/openpi"
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH="$ENVDIR/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="$OPENPI/.torch_cache"
export PYTHONPATH="$OPENPI/src:$OPENPI/third_party/libero:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
```

---

## 常见问题

| 问题 | 解决 |
|------|------|
| `glGetError` AttributeError | 缺 libOSMesa.so 或 `LD_LIBRARY_PATH` 未设 |
| `eglQueryString` AttributeError | 设 `MUJOCO_GL=osmesa` |
| `No module named 'matplotlib'` | `pip install matplotlib` |
| `QOSMaxJobsPerUserLimit` | 先 scancel 一个任务 |
| `QOSMaxGRESPerUser` | GPU 配额满，换型号或等待 |
| torch.load error | 第 5.2 节 patch（weights_only=False） |
| V100 崩溃 | cu128 不支持 sm_70，排除 V100 |

---
---

# 附录：环境从零搭建

> 以下仅在需要重建环境时参考。当前环境已搭建完成。

## A1. 克隆代码

```bash
cd /projects/chuanlia001ssd/repos
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/CharlieXie/openpi.git
cd openpi && git checkout v1.2_e2e
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

## A2. 创建 Conda 环境

```bash
module load Miniforge3 && source activate
export TMPDIR=/projects/chuanlia001ssd/tmp
export PIP_CACHE_DIR=/projects/chuanlia001ssd/tmp/pip-cache
mkdir -p $TMPDIR $PIP_CACHE_DIR
conda create -p /home/chuanlia001/envs/openpi python=3.11 mesalib -c conda-forge -y
```

## A3. 安装 libOSMesa（conda mesalib 不含，必须手动提取）

```bash
mkdir -p /tmp/osmesa-extract && cd /tmp/osmesa-extract
apt-get download libosmesa6 && apt-get download libglapi-mesa
dpkg-deb -x libosmesa6_*.deb extracted/
dpkg-deb -x libglapi-mesa_*.deb extracted/
LIBDIR=/home/chuanlia001/envs/openpi/lib
cp extracted/usr/lib/x86_64-linux-gnu/libOSMesa.so.8.0.0 "$LIBDIR/"
cp extracted/usr/lib/x86_64-linux-gnu/libglapi.so.0.0.0 "$LIBDIR/"
cd "$LIBDIR"
ln -sf libOSMesa.so.8.0.0 libOSMesa.so.8
ln -sf libOSMesa.so.8 libOSMesa.so.6
ln -sf libOSMesa.so.8 libOSMesa.so
ln -sf libglapi.so.0.0.0 libglapi.so.0
ln -sf libglapi.so.0 libglapi.so
rm -rf /tmp/osmesa-extract
```

## A4. 安装 Python 包

```bash
PYTHON=/home/chuanlia001/envs/openpi/bin/python
PIP="$PYTHON -m pip"

# PyTorch
$PIP install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# JAX CPU-only（仅 import 需要）
$PIP install jax==0.5.3 jaxlib==0.5.3

# 核心依赖
$PIP install \
    "transformers==4.53.2" "flax==0.10.2" "safetensors" "sentencepiece>=0.2.0" \
    "orbax-checkpoint==0.11.13" "augmax>=0.3.4" "einops>=0.8.0" \
    "beartype==0.19.0" "jaxtyping==0.2.36" \
    "imageio>=2.36.1" "imageio-ffmpeg" "pillow>=11.0.0" "numpy>=1.22.4,<2" \
    "pyyaml" "fsspec[gcs]>=2024.6.0" "filelock>=3.16.1" "tqdm-loggable>=0.2" \
    "typing-extensions>=4.12.2" "ml_collections==1.0.0" "dm-tree>=0.1.8" \
    "pytest" "equinox>=0.11.8" "rich>=14.0.0" \
    "opencv-python>=4.10.0.84" "numpydantic>=1.6.6" "polars>=1.30.0"

# Eval 依赖
$PIP install robosuite==1.4.1 transforms3d bddl easydict "gym==0.26.2" matplotlib

# LIBERO
$PIP install -e /projects/chuanlia001ssd/repos/openpi/third_party/libero
```

## A5. 打 Patch

```bash
OPENPI=/projects/chuanlia001ssd/repos/openpi

# transformers 补丁
cp -r $OPENPI/src/openpi/models_pytorch/transformers_replace/* \
    /home/chuanlia001/envs/openpi/lib/python3.11/site-packages/transformers/

# LIBERO torch.load 兼容 PyTorch 2.6+
LIBERO_INIT=$OPENPI/third_party/libero/libero/libero/benchmark/__init__.py
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' "$LIBERO_INIT"
```

## A6. 验证环境

```bash
OPENPI=/projects/chuanlia001ssd/repos/openpi
ENVDIR=/home/chuanlia001/envs/openpi
MUJOCO_GL=osmesa LD_LIBRARY_PATH=$ENVDIR/lib:${LD_LIBRARY_PATH:-} \
PYTHONPATH=$OPENPI/src:$OPENPI/third_party/libero:${PYTHONPATH:-} \
$ENVDIR/bin/python -c "
from openpi.waypoint.normalize import NormalizationHelper, load_dataset_statistics
from openpi.waypoint.robot_config import get_robot_config
from openpi.waypoint.tokenizer import WaypointTokenizer
from openpi.waypoint.joint_model import PI0WaypointJoint
from libero.libero import benchmark
import robosuite
from OpenGL.GL import glGetError
print('All imports OK!')
"
```

## A7. 训练额外依赖（在 eval 环境基础上）

详见 `NTU_CLUSTER_TRAIN_RUNBOOK.md`。简要：

```bash
$PIP install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$PIP install wandb "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"
$PIP install "ml_dtypes>=0.4.0"           # 修复 TF 2.15 降级冲突
$PIP install "nvidia-nccl-cu12>=2.29"     # Blackwell 多卡 DDP 必须
```
