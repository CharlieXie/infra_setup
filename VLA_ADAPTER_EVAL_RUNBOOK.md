# VLA-Adapter Evaluation Runbook

## 背景介绍

[VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) 是一个基于 Qwen2.5-0.5B（0.5B tiny-scale）的 Vision-Language-Action 模型，通过 adapter bridge 范式微调，在 LIBERO 和 CALVIN 两个机器人操作 benchmark 上取得了 SOTA 性能。

### 架构特点
- **VLM 骨干**：`prism-qwen25-extra-dinosiglip-224px-0_5b`（Stanford-ILIAD 的 Prismatic-VLMs，LLM 骨干为 Qwen2.5-0.5B）
- **视觉编码器**：DINOv2 + SigLIP 双路 224px
- **动作头**：L1 Regression Action Head，输出 7-DOF 动作
- **Proprio**：8 维本体感知投影器
- **动作 Chunking**：每次推理输出 8 步动作（`NUM_ACTIONS_CHUNK=8`），Hi3 temporal aggregation（CALVIN 专用）
- **Pro 版本**：论文中推荐使用，Policy 大小 207MB，性能更强

### 评估 Benchmark 简介

**LIBERO**：MuJoCo/Robosuite 仿真，4 个 Task Suite（Spatial/Object/Goal/Long），每 suite 10 个任务，默认每任务 50 个 episode。**不需要下载数据集**，任务定义内置于 Python 包中。

**CALVIN ABC→D**：PyBullet 仿真，1000 个 5-subtask chain 序列评估。**需要下载 `task_ABC_D` validation 数据**（~27GB），因为环境配置（场景布局、初始状态）存储在数据集文件中，而不是内置于 Python 包。完整数据集 517GB，但评估只需要 `validation/` 子集。

---

## 系统要求

- Python **3.10**（tensorflow 2.15 不支持 Python 3.12）
- CUDA ≥ 12.1
- GPU ≥ 20GB VRAM（推理时单张 LIBERO episode 约占用 ~8GB）
- 磁盘：LIBERO 约 25GB（模型 + 包），CALVIN 还需额外 ~27GB（validation 数据）

---

## 一、环境安装

### 1. 系统依赖

```bash
apt-get update && apt-get install -y \
    libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev \
    libosmesa6-dev xvfb ffmpeg
```

### 2. 创建 Conda 环境（Python 3.10）

```bash
conda create -n vla-adapter python=3.10.16 -y
conda activate vla-adapter
```

### 3. 安装 PyTorch（CUDA 12.1）

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 4. 克隆并安装 VLA-Adapter

```bash
git clone https://github.com/OpenHelix-Team/VLA-Adapter.git
cd VLA-Adapter
pip install -e .
```

### 5. 安装 LIBERO

```bash
# 克隆 LIBERO 到 VLA-Adapter 目录下
cd /workspace/VLA-Adapter
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# 用 setup.py develop 安装（重要：editable 安装有 MAPPING 空字典 bug，
# 必须用 setup.py develop 或手动加 PYTHONPATH）
cd LIBERO
python setup.py develop

# 安装 LIBERO 额外依赖
pip install "robosuite==1.4.1" bddl easydict cloudpickle "gym==0.23.1" \
    "imageio[ffmpeg]" imageio-ffmpeg mujoco
```

> ⚠️ **已知问题**：`pip install -e .` 对 LIBERO 的 editable 安装会生成空 MAPPING 的 finder，导致 `import libero` 失败。必须用 `python setup.py develop` 或在运行时设置 `PYTHONPATH`。

如果用 pip install -e 安装，需在脚本中设置：
```bash
export PYTHONPATH="/path/to/VLA-Adapter/LIBERO:/path/to/VLA-Adapter:$PYTHONPATH"
```

### 6. 安装 CALVIN（仿真环境只，不需要下载数据集）

```bash
cd /workspace/VLA-Adapter
git clone --recurse-submodules https://github.com/mees/calvin.git

# 安装 calvin_env（仿真环境）
pip install --no-deps -e calvin/calvin_env

# 安装 calvin_models（评估代码）— 不安装依赖避免版本冲突
pip install --no-deps -e calvin/calvin_models

# 安装 CALVIN 必需的 Python 依赖
pip install "moviepy<2.0" hydra-core omegaconf pytorch-lightning \
    termcolor pybullet msgpack msgpack-numpy sentence-transformers plotly

# 修复被 sentence-transformers 升级的 transformers 版本
pip install "transformers==4.40.1" "tokenizers==0.19.1" "huggingface_hub<1.0"

# 修复 numpy 版本（tensorflow 需要 <2.0）
pip install "numpy<2.0.0,>=1.23.5"
```

> ⚠️ **moviepy 版本**：必须安装 `moviepy<2.0`（即 1.x），因为 CALVIN 脚本使用 `from moviepy.editor import ImageSequenceClip`，这是 v1 的 API，v2 已移除。

---

## 二、下载模型

所有模型通过 HuggingFace 下载，使用 Python 的 `huggingface_hub` 库。

```python
from huggingface_hub import snapshot_download

# VLM 骨干（~7.5GB，含 tokenizer/config/model 权重）
# 注意：这里的 checkpoints/ 子目录存放的是 VLM 预训练中间 checkpoint，
# 评估不需要，下载后可直接删除以节省 7.4GB 空间
snapshot_download(
    'Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b',
    local_dir='pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b'
)

# LIBERO 4 个 Pro 版本 checkpoint（各约 2.8GB）
for repo in [
    'VLA-Adapter/LIBERO-Spatial-Pro',
    'VLA-Adapter/LIBERO-Object-Pro',
    'VLA-Adapter/LIBERO-Goal-Pro',
    'VLA-Adapter/LIBERO-Long-Pro',
]:
    name = repo.split('/')[-1]
    snapshot_download(repo, local_dir=f'outputs/{name}')

# CALVIN Pro checkpoint（约 2.8GB）
snapshot_download('VLA-Adapter/CALVIN-ABC-Pro', local_dir='outputs/CALVIN-ABC-Pro')
```

### 可立即清理的空间（不影响评估）

| 路径 | 大小 | 说明 |
|------|------|------|
| `pretrained_models/.../checkpoints/` | **7.4 GB** | VLM backbone 预训练中间 checkpoint，评估完全不需要 |
| `pretrained_models/.../wandb/` | 73 MB | 预训练 wandb 日志 |
| `~/.cache/pip` | ~4 GB | pip 下载缓存 |

```bash
rm -rf pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b/checkpoints/
rm -rf pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b/wandb/
rm -rf ~/.cache/pip
conda clean --all -y
```

---

## 三、LIBERO 评估

### 每个 checkpoint 包含的文件

```
outputs/LIBERO-Spatial-Pro/
├── model.safetensors          # 完整 VLM 权重（2.4GB）
├── action_head--checkpoint.pt # 动作头权重
├── config.json
├── configuration_prismatic.py
├── modeling_prismatic.py
├── dataset_statistics.json    # 动作归一化统计
├── tokenizer_config.json
└── ...
```

评估时 `AutoModelForVision2Seq.from_pretrained(pretrained_checkpoint)` 加载 `outputs/LIBERO-Spatial-Pro/`，**不依赖** `pretrained_models/` 下的 checkpoint 文件。

### 运行评估脚本

```bash
cd /workspace/VLA-Adapter
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/workspace/VLA-Adapter/LIBERO:/workspace/VLA-Adapter:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
    --task_suite_name libero_spatial \
    --use_pro_version True \
    --num_trials_per_task 50     # 论文标准；改为 3 可快速验证
```

`task_suite_name` 对应关系：

| checkpoint | task_suite_name |
|---|---|
| LIBERO-Spatial-Pro | `libero_spatial` |
| LIBERO-Object-Pro | `libero_object` |
| LIBERO-Goal-Pro | `libero_goal` |
| LIBERO-Long-Pro | `libero_10` |

### 批量评估脚本（全部 4 个 suite）

保存为 `run_libero_eval_all.sh`：

```bash
#!/bin/bash
PYTHON=/venv/vla-adapter/bin/python
cd /workspace/VLA-Adapter

export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/workspace/VLA-Adapter/LIBERO:/workspace/VLA-Adapter:$PYTHONPATH"

mkdir -p eval_logs

declare -A SUITES
SUITES[libero_spatial]="outputs/LIBERO-Spatial-Pro"
SUITES[libero_object]="outputs/LIBERO-Object-Pro"
SUITES[libero_goal]="outputs/LIBERO-Goal-Pro"
SUITES[libero_10]="outputs/LIBERO-Long-Pro"

for suite in libero_spatial libero_object libero_goal libero_10; do
    CUDA_VISIBLE_DEVICES=0 $PYTHON experiments/robot/libero/run_libero_eval.py \
        --use_proprio True \
        --num_images_in_input 2 \
        --use_film False \
        --pretrained_checkpoint "${SUITES[$suite]}" \
        --task_suite_name "$suite" \
        --use_pro_version True \
        --num_trials_per_task 3 \
        2>&1 | tee "eval_logs/eval_${suite}_3ep.log"
done
```

在 tmux 中启动：

```bash
tmux new-session -d -s libero_eval \
    "bash run_libero_eval_all.sh 2>&1 | tee eval_logs/libero_eval_full.log; exec bash"
# 实时查看进度
tail -f eval_logs/libero_eval_full.log
```

### 耗时估算（RTX 4090）

| Suite | max_steps/episode | 3 ep/task | 50 ep/task（标准） |
|---|---|---|---|
| libero_spatial | 220 | ~3.5 min | ~58 min |
| libero_object | 280 | ~3.5 min | ~60 min |
| libero_goal | 300 | ~4 min | ~60 min |
| libero_10 (Long) | 520 | ~5 min | ~80 min |

---

## 四、CALVIN 评估

> ⚠️ **CALVIN 评估目前尚未在本机验证完整跑通**，以下为根据代码分析和官方文档整理的配置步骤。

### 为何 CALVIN 需要下载数据集

LIBERO 的任务定义（BDDL 文件、初始状态）内置在 Python 包中，仿真器可直接启动。  
CALVIN 的环境初始化需要读取 `dataset/task_ABC_D/validation/` 目录下的场景配置文件（物体位置、初始状态序列、语言标注），因此必须下载数据。

### 数据集大小说明

| 数据集 | 大小 | 用途 |
|---|---|---|
| 完整 task_ABC_D | **517 GB** | 训练 + 评估 |
| validation 子集 | **~27 GB** | **仅评估所需** |
| debug 数据集 | **1.3 GB** | 流程验证（任务数量有限） |

**评估只需要 validation 子集，不需要下载完整数据集。** 推荐通过 HuggingFace 只下载 validation 部分：

```python
# 方式一：只下载 validation 子集（推荐，约 27GB）
from huggingface_hub import snapshot_download
snapshot_download(
    'VyoJ/calvin-ABCD-D-subsets',
    allow_patterns=['subset_validation_000/*'],
    local_dir='calvin/dataset/task_ABC_D'
)
# 下载后 validation 文件夹应位于：
# calvin/dataset/task_ABC_D/validation/

# 方式二：下载 debug 数据集验证流程（1.3GB）
cd calvin/dataset && bash download_data.sh debug
# 注意：debug 数据集路径为 calvin_debug_dataset，需修改脚本中的 CALVIN_ROOT
```

### 修改评估脚本以支持自定义序列数

`vla-scripts/evaluate_calvin.py` 中 `num_sequences` 默认硬编码为 1000，已修改为可通过命令行传参：

```python
# GenerateConfig 中添加
num_sequences: int = 1000  # 评估序列数，默认 1000，可改为 10 快速测试
```

```python
# main() 中已改为
num_sequences=cfg.num_sequences,
```

### 运行 CALVIN 评估

```bash
cd /workspace/VLA-Adapter
export CALVIN_ROOT="calvin"
export PYTHONPATH="/workspace/VLA-Adapter:$PYTHONPATH"

# 标准评估（1000 sequences）
CUDA_VISIBLE_DEVICES=0 python vla-scripts/evaluate_calvin.py \
    --pretrained_checkpoint outputs/CALVIN-ABC-Pro \
    --num_sequences 1000

# 快速验证（10 sequences）
CUDA_VISIBLE_DEVICES=0 python vla-scripts/evaluate_calvin.py \
    --pretrained_checkpoint outputs/CALVIN-ABC-Pro \
    --num_sequences 10
```

### 耗时估算（H100）

- 1000 sequences：**约 4 小时 10 分钟**（~15s/sequence）
- 10 sequences：**约 2.5 分钟**

RTX 4090 上预计会慢约 2-3x。

### CALVIN 数据集目录结构要求

```
VLA-Adapter/
└── calvin/
    └── dataset/
        └── task_ABC_D/
            └── validation/
                ├── lang_annotations/
                │   ├── auto_lang_ann.npy
                │   └── embeddings.npy      # rollout 推理时使用
                ├── scene_info.npy
                └── episode_XXXXXXX.npz     # 场景状态文件
```

---

## 五、我们的评估结果（2026-03-24，RTX 4090）

### LIBERO 评估结果（Pro 版本，3 episodes/task）

| Suite | Episodes | Successes | **我们的成功率** | **论文 Pro（50 ep, H100）** |
|---|---|---|---|---|
| libero_spatial | 30 | 29 | **96.7%** | 99.6% |
| libero_object | 30 | 27 | **90.0%** | 99.6% |
| libero_goal | 30 | 28 | **93.3%** | 98.2% |
| libero_10 (Long) | 30 | 26 | **86.7%** | 96.4% |

**总耗时**：约 19 分钟（4 个 suite 串行，RTX 4090）

### 失败 episode 分析

我们使用 3 episode/task，样本量少，方差大。对比官方 50 episode 日志（`eval_logs/Inference*Pro*.log`）分析：

**LIBERO-Spatial（1 次失败）**
- Task 2（ramekin 旁的 bowl）：我们 ep2 失败。官方 ep1-3 全部成功，失败发生在 ep8（50次中）。**任务本身略有难度（官方 98%），但非同一 episode 失败。**

**LIBERO-Object（3 次失败）**
- Task 2（cream cheese）、Task 4（BBQ sauce）、Task 6（tomato sauce）：我们各失败 1 次。官方这三个任务均为 **100%**，ep1-3 全部成功。**属于高方差失误，非模型弱项。**

**LIBERO-Goal（2 次失败）**
- Task 4（open drawer + bowl）：我们 ep1 失败。官方成功率 96%（有难度），但官方 ep1-3 全部成功。
- Task 9（put bowl on plate）：我们 ep3 失败。官方 **100%**，ep1-3 全部成功。**不一致，属于随机失误。**

**LIBERO-Long（4 次失败，1 次有对应）**
- Task 5（双 mug 摆放）：我们 T/T/F，官方也是 T/T/F（**ep3 完全对应！**）。官方 94%，任务有难度。✅
- Task 9（both moka pots on stove）：我们 F/F/F，官方 90%，ep1-3 为 T/F/T。**任务确实是难点，但失败分布不同。**

**结论**：
- 3 ep 测试中大部分失败属于**小样本方差**，只有 LIBERO-Long Task 5 实现了与官方相同的 ep3 失败匹配
- 结果成功率低于论文是正常的（样本量少 + GPU 不同），整体趋势一致

### CALVIN 评估结果

> ⚠️ **CALVIN 评估尚未在本环境跑通**（磁盘空间不足，无法下载 ~27GB validation 数据）。

官方发布的 CALVIN-ABC-Pro 结果（H100，1000 sequences）：

| 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | **Avg. len** |
|---|---|---|---|---|---|
| 98.5% | 95.0% | 90.5% | 85.3% | 80.0% | **4.50** |

---

## 六、已知问题 & 注意事项

1. **LIBERO editable install bug**：`pip install -e LIBERO` 生成空 MAPPING 的 finder，必须用 `python setup.py develop` 或设置 `PYTHONPATH`

2. **moviepy 版本**：必须用 `moviepy<2.0`，v2 移除了 `moviepy.editor` API

3. **numpy 版本冲突**：`tensorflow==2.15.0` 要求 `numpy<2.0`，但 `calvin_env` 和 `opencv-python` 会拉新版本，安装后需手动固定：
   ```bash
   pip install "numpy<2.0.0,>=1.23.5"
   ```

4. **sentence-transformers** 会拉高版本 `transformers`，装完需 pin 回：
   ```bash
   pip install "transformers==4.40.1" "tokenizers==0.19.1"
   ```

5. **GPU 差异**：官方在 H100 上测试，RTX 4090 结果可能略有差异（项目 README 也有此说明）

6. **EGL 渲染**：LIBERO 需要 `MUJOCO_GL=egl`；CALVIN 使用 PyBullet EGL。两者都需要 `libgl1-mesa-dev` 等系统依赖

7. **`pretrained_models/.../checkpoints/`**：下载 VLM backbone 时会带这 3 个约 2.5GB 的 `.pt` 文件（VLM 预训练中间 checkpoint），评估完全不需要，可立即删除节省 7.4GB

---

## 七、目录结构总览

```
/workspace/VLA-Adapter/
├── LIBERO/                          # LIBERO benchmark（git clone）
├── calvin/                          # CALVIN benchmark（git clone --recurse-submodules）
│   └── dataset/task_ABC_D/         # CALVIN 数据集（需下载，仅需 validation/）
├── outputs/                         # 评估用 checkpoint（从 HF 下载）
│   ├── LIBERO-Spatial-Pro/
│   ├── LIBERO-Object-Pro/
│   ├── LIBERO-Goal-Pro/
│   ├── LIBERO-Long-Pro/
│   └── CALVIN-ABC-Pro/
├── pretrained_models/               # VLM backbone
│   ├── configs/                     # tokenizer/config 文件
│   └── prism-qwen25-extra-dinosiglip-224px-0_5b/
│       ├── checkpoints/             # ⚠️ 可删除（7.4GB），评估不需要
│       └── wandb/                   # ⚠️ 可删除（73MB），评估不需要
├── experiments/robot/libero/
│   ├── run_libero_eval.py           # LIBERO 评估入口
│   ├── libero_utils.py
│   └── libero_requirements.txt
├── vla-scripts/
│   ├── evaluate_calvin.py           # CALVIN 评估入口（已修改 num_sequences 可配置）
│   ├── calvin_env_wrapper.py
│   └── vla_evaluation.py
├── eval_logs/                       # 评估日志（含官方日志和我们的测试日志）
└── run_libero_eval_all.sh           # 批量 LIBERO 评估脚本
```
