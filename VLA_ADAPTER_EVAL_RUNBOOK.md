# VLA-Adapter Evaluation Runbook

> **目标读者**：AI coding agent。本文档用于在全新机器上从零配置 VLA-Adapter 环境并运行 LIBERO / CALVIN 评估。

## 前置条件

- Linux 系统，CUDA >= 12.1，GPU >= 20GB VRAM
- `conda` 已安装且在 PATH 中（如不在，检查 `/opt/miniforge3/bin/conda` 或 `/opt/miniconda3/bin/conda`，将其加入 PATH）
- 磁盘空间：LIBERO 约 25GB，CALVIN 额外 ~30GB（数据 27GB + 解压临时空间）
- 工作目录：`/workspace`

---

## 一键安装脚本

将以下脚本保存为 `/workspace/setup_vla_adapter.sh` 并执行。脚本是幂等的，重复运行会跳过已完成步骤。

```bash
#!/bin/bash
set -euo pipefail

WORKDIR="/workspace"
VENV="/venv/vla-adapter"
CONDA_ENV="vla-adapter"
PYTHON_VERSION="3.10.16"

cd "$WORKDIR"

########################################
# 1. 系统依赖
########################################
echo ">>> [1/9] 安装系统依赖..."
apt-get update -qq && apt-get install -y -qq \
    libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev \
    libosmesa6-dev xvfb ffmpeg > /dev/null 2>&1
echo "    OK"

########################################
# 2. Conda 环境（Python 3.10 是硬性要求，tensorflow 2.15 不支持 3.12）
########################################
echo ">>> [2/9] 创建 conda 环境 $CONDA_ENV (Python $PYTHON_VERSION)..."
if conda env list | grep -q "$CONDA_ENV"; then
    echo "    已存在，跳过"
else
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y -q
fi

# 获取 conda 环境中的 python/pip 路径
CONDA_PREFIX=$(conda env list | grep "$CONDA_ENV" | awk '{print $NF}')
PY="$CONDA_PREFIX/bin/python"
PIP="$CONDA_PREFIX/bin/pip"
echo "    Python: $($PY --version)"

########################################
# 3. PyTorch（CUDA 12.1 wheels）
########################################
echo ">>> [3/9] 安装 PyTorch..."
if $PY -c "import torch; assert 'cu121' in torch.__version__" 2>/dev/null; then
    echo "    已安装，跳过"
else
    $PIP install -q torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu121
fi
echo "    OK: $($PY -c 'import torch; print(torch.__version__)')"

########################################
# 4. 克隆并安装 VLA-Adapter
########################################
echo ">>> [4/9] 安装 VLA-Adapter..."
cd "$WORKDIR"
[ -d VLA-Adapter ] || git clone -q https://github.com/OpenHelix-Team/VLA-Adapter.git
cd VLA-Adapter
$PIP install -q -e .
echo "    OK"

########################################
# 5. 安装 LIBERO
# ⚠️ 必须用 setup.py develop，pip install -e 会导致 import libero 失败
#    （生成空 MAPPING 的 finder，是 LIBERO 包自身的 bug）
########################################
echo ">>> [5/9] 安装 LIBERO..."
cd "$WORKDIR/VLA-Adapter"
[ -d LIBERO ] || git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && $PY setup.py develop -q 2>/dev/null
$PIP install -q "robosuite==1.4.1" bddl easydict cloudpickle "gym==0.23.1" \
    "imageio[ffmpeg]" imageio-ffmpeg mujoco
echo "    OK"

########################################
# 6. 安装 CALVIN
# 安装顺序很重要：先装包 → 装依赖 → 用 conda 装 pyhash → pin 版本
########################################
echo ">>> [6/9] 安装 CALVIN..."
cd "$WORKDIR/VLA-Adapter"
[ -d calvin ] || git clone -q --recurse-submodules https://github.com/mees/calvin.git
$PIP install -q --no-deps -e calvin/calvin_env
$PIP install -q --no-deps -e calvin/calvin_models

# CALVIN Python 依赖
# ⚠️ moviepy 必须 <2.0（v2 移除了 moviepy.editor API，CALVIN 脚本会 import 失败）
$PIP install -q "moviepy<2.0" hydra-core omegaconf pytorch-lightning \
    termcolor pybullet msgpack msgpack-numpy sentence-transformers plotly \
    hydra-colorlog numpy-quaternion pandas

# ⚠️ pyhash：pip install 在 Python 3.10+ 会报 "use_2to3 is invalid"，只能用 conda
if ! $PY -c "import pyhash" 2>/dev/null; then
    conda install -n "$CONDA_ENV" -c conda-forge pyhash -y -q 2>/dev/null
fi

# ⚠️ sentence-transformers 会把 transformers 升级到 5.x，必须 pin 回来
$PIP install -q "transformers==4.40.1" "tokenizers==0.19.1" "huggingface_hub<1.0"
# ⚠️ opencv-python/mujoco 会把 numpy 升级到 2.x，tensorflow 2.15 要求 <2.0
$PIP install -q "numpy<2.0.0,>=1.23.5"

echo "    OK"

########################################
# 7. 验证关键依赖版本
########################################
echo ">>> [7/9] 验证依赖..."
$PY -c "
import torch, transformers, numpy, pyhash
assert '2.2.0' in torch.__version__, f'torch version mismatch: {torch.__version__}'
assert transformers.__version__ == '4.40.1', f'transformers version mismatch: {transformers.__version__}'
assert numpy.__version__.startswith('1.'), f'numpy version mismatch: {numpy.__version__}'
assert torch.cuda.is_available(), 'CUDA not available'
print(f'    torch={torch.__version__}, transformers={transformers.__version__}, numpy={numpy.__version__}, CUDA=OK')
"

########################################
# 8. 下载模型（HuggingFace）
########################################
echo ">>> [8/9] 下载模型..."

# 可选加速：hf_transfer 多线程下载
$PIP install -q hf_transfer 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1

$PY -c "
from huggingface_hub import snapshot_download
import os

models = {
    # VLM backbone (~7.5GB)
    'Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b': 'pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b',
    # LIBERO checkpoints (各 ~2.8GB)
    'VLA-Adapter/LIBERO-Spatial-Pro': 'outputs/LIBERO-Spatial-Pro',
    'VLA-Adapter/LIBERO-Object-Pro': 'outputs/LIBERO-Object-Pro',
    'VLA-Adapter/LIBERO-Goal-Pro':   'outputs/LIBERO-Goal-Pro',
    'VLA-Adapter/LIBERO-Long-Pro':   'outputs/LIBERO-Long-Pro',
    # CALVIN checkpoint (~2.8GB)
    'VLA-Adapter/CALVIN-ABC-Pro':    'outputs/CALVIN-ABC-Pro',
}

os.chdir('$WORKDIR/VLA-Adapter')
for repo, local_dir in models.items():
    if os.path.isdir(local_dir) and len(os.listdir(local_dir)) > 3:
        print(f'    {repo.split(\"/\")[-1]}: 已存在，跳过')
        continue
    print(f'    下载 {repo.split(\"/\")[-1]}...')
    snapshot_download(repo, local_dir=local_dir)
print('    模型下载完成')
"

# 清理不需要的文件（节省 ~7.5GB）
rm -rf "$WORKDIR/VLA-Adapter/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b/checkpoints/"
rm -rf "$WORKDIR/VLA-Adapter/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b/wandb/"
echo "    OK"

########################################
# 9. 下载 CALVIN 数据集 + hydra 配置
########################################
echo ">>> [9/9] 下载 CALVIN 数据集..."
CALVIN_VAL_DIR="$WORKDIR/VLA-Adapter/calvin/dataset/task_ABC_D/validation"

if [ -d "$CALVIN_VAL_DIR" ] && [ -f "$CALVIN_VAL_DIR/.hydra/merged_config.yaml" ]; then
    echo "    已存在，跳过"
else
    cd "$WORKDIR/VLA-Adapter"

    # 下载 validation zip (~27GB)
    # ⚠️ 必须指定 repo_type='dataset'，否则报 401（默认按 model 仓库查找）
    # ⚠️ 实际文件是 zip，不能用 snapshot_download + allow_patterns（会匹配 0 文件）
    $PY -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    'VyoJ/calvin-ABCD-D-subsets',
    repo_type='dataset',
    filename='validation/subset_validation_000.zip',
    local_dir='calvin/dataset/task_ABC_D_download'
)
print('    zip 下载完成')
"

    # 解压并整理目录
    # ⚠️ zip 内部路径是 subset_validation_000/validation/，需要移到 task_ABC_D/validation/
    mkdir -p calvin/dataset/task_ABC_D
    unzip -q -o calvin/dataset/task_ABC_D_download/validation/subset_validation_000.zip \
        -d calvin/dataset/task_ABC_D/
    mv calvin/dataset/task_ABC_D/subset_validation_000/validation "$CALVIN_VAL_DIR"
    rm -rf calvin/dataset/task_ABC_D/subset_validation_000 calvin/dataset/task_ABC_D_download

    # ⚠️ 关键文件：.hydra/merged_config.yaml
    # HuggingFace subset 不含此文件，但 CALVIN 环境初始化必须读取它
    # (play_table_env.py:275 的 OmegaConf.load)
    # 来源：https://github.com/OpenHelix-Team/VLA-Adapter/issues/25
    mkdir -p "$CALVIN_VAL_DIR/.hydra"
    wget -q -O "$CALVIN_VAL_DIR/.hydra/merged_config.yaml" \
        "https://github.com/user-attachments/files/23740664/merged_config.yaml"
fi
echo "    OK"

########################################
# 10. Patch evaluate_calvin.py（支持 --num_sequences 参数）
########################################
echo ">>> Patch evaluate_calvin.py..."
EVAL_SCRIPT="$WORKDIR/VLA-Adapter/vla-scripts/evaluate_calvin.py"

# 添加 num_sequences 字段到 GenerateConfig
if ! grep -q "num_sequences:" "$EVAL_SCRIPT"; then
    sed -i '/initial_states_path.*DEFAULT/a\    num_sequences: int = 1000' "$EVAL_SCRIPT"
fi

# 将硬编码的 num_sequences=1000 改为 cfg.num_sequences
sed -i 's/num_sequences=1000/num_sequences=cfg.num_sequences/' "$EVAL_SCRIPT"

echo "    OK"
echo ""
echo "========================================="
echo " 安装完成！可以开始评估。"
echo "========================================="
```

执行方式：

```bash
chmod +x /workspace/setup_vla_adapter.sh
bash /workspace/setup_vla_adapter.sh 2>&1 | tee /workspace/setup.log
```

验证安装成功的标志：脚本最后输出 `安装完成！可以开始评估。` 且无 `set -e` 导致的中途退出。

---

## 运行评估

### LIBERO 评估

```bash
CONDA_PREFIX=$(conda env list | grep vla-adapter | awk '{print $NF}')
PY="$CONDA_PREFIX/bin/python"
cd /workspace/VLA-Adapter

export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/workspace/VLA-Adapter/LIBERO:/workspace/VLA-Adapter:$PYTHONPATH"
```

单个 suite 评估：

```bash
CUDA_VISIBLE_DEVICES=0 $PY experiments/robot/libero/run_libero_eval.py \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
    --task_suite_name libero_spatial \
    --use_pro_version True \
    --num_trials_per_task 3
```

checkpoint 与 task_suite_name 对应关系：

| checkpoint | task_suite_name |
|---|---|
| `outputs/LIBERO-Spatial-Pro` | `libero_spatial` |
| `outputs/LIBERO-Object-Pro` | `libero_object` |
| `outputs/LIBERO-Goal-Pro` | `libero_goal` |
| `outputs/LIBERO-Long-Pro` | `libero_10` |

批量评估全部 4 个 suite（`--num_trials_per_task 3` 快速验证，改 50 为论文标准）：

```bash
cd /workspace/VLA-Adapter
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/workspace/VLA-Adapter/LIBERO:/workspace/VLA-Adapter:$PYTHONPATH"

CONDA_PREFIX=$(conda env list | grep vla-adapter | awk '{print $NF}')
PY="$CONDA_PREFIX/bin/python"

for suite_ckpt in \
    "libero_spatial:outputs/LIBERO-Spatial-Pro" \
    "libero_object:outputs/LIBERO-Object-Pro" \
    "libero_goal:outputs/LIBERO-Goal-Pro" \
    "libero_10:outputs/LIBERO-Long-Pro"; do

    suite="${suite_ckpt%%:*}"
    ckpt="${suite_ckpt#*:}"
    echo "=== Evaluating $suite ==="
    CUDA_VISIBLE_DEVICES=0 $PY experiments/robot/libero/run_libero_eval.py \
        --use_proprio True \
        --num_images_in_input 2 \
        --use_film False \
        --pretrained_checkpoint "$ckpt" \
        --task_suite_name "$suite" \
        --use_pro_version True \
        --num_trials_per_task 3
done
```

耗时估算（RTX 4090，`--num_trials_per_task 3`）：

| Suite | 3 ep/task | 50 ep/task |
|---|---|---|
| libero_spatial | ~3.5 min | ~58 min |
| libero_object | ~3.5 min | ~60 min |
| libero_goal | ~4 min | ~60 min |
| libero_10 | ~5 min | ~80 min |
| **全部 4 suite** | **~16 min** | **~4.3 h** |

### CALVIN 评估

```bash
CONDA_PREFIX=$(conda env list | grep vla-adapter | awk '{print $NF}')
PY="$CONDA_PREFIX/bin/python"
cd /workspace/VLA-Adapter

export CALVIN_ROOT="calvin"
export PYTHONPATH="/workspace/VLA-Adapter:$PYTHONPATH"

# 快速验证（10 sequences，约 3 分钟）
CUDA_VISIBLE_DEVICES=0 $PY vla-scripts/evaluate_calvin.py \
    --pretrained_checkpoint outputs/CALVIN-ABC-Pro \
    --num_sequences 10

# 标准评估（1000 sequences，约 4.5 小时）
CUDA_VISIBLE_DEVICES=0 $PY vla-scripts/evaluate_calvin.py \
    --pretrained_checkpoint outputs/CALVIN-ABC-Pro \
    --num_sequences 1000
```

实时查看进度（评估过程中 stdout 可能被缓冲）：

```bash
# 查看最新结果文件
ls -t evaluation_results/calvin/*/success_rate.txt | head -1 | xargs tail -f
```

耗时估算：

| 环境 | 10 sequences | 1000 sequences |
|---|---|---|
| RTX 4090（实测） | ~2 min 50s (~17s/seq) | ~4h 40min |
| H100（官方） | ~2 min 30s (~15s/seq) | ~4h 10min |

---

## 预期结果

用于判断评估是否正常运行。小样本（3 ep / 10 seq）方差较大，低于下列范围说明环境可能有问题。

### LIBERO（3 episodes/task，Pro 版本）

| Suite | 预期成功率范围 | 论文标准（50 ep, H100） |
|---|---|---|
| libero_spatial | 85-100% | 99.6% |
| libero_object | 80-100% | 99.6% |
| libero_goal | 80-100% | 98.2% |
| libero_10 | 70-100% | 96.4% |

### CALVIN（10 sequences，Pro 版本）

| 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg. len |
|---|---|---|---|---|---|
| 90-100% | 80-100% | 70-100% | 60-100% | 50-100% | >= 4.0 |

论文标准（1000 seq, H100）：98.5% / 95.0% / 90.5% / 85.3% / 80.0% / Avg 4.50

---

## Troubleshooting

按错误信息查找修复方法。

| 错误信息 | 原因 | 修复 |
|---|---|---|
| `ModuleNotFoundError: No module named 'pyhash'` | pip 安装失败（Python 3.10 不兼容） | `conda install -c conda-forge pyhash -y` |
| `FileNotFoundError: .../validation/.hydra/merged_config.yaml` | HuggingFace subset 数据集不含此文件 | `mkdir -p calvin/dataset/task_ABC_D/validation/.hydra && wget -O calvin/dataset/task_ABC_D/validation/.hydra/merged_config.yaml "https://github.com/user-attachments/files/23740664/merged_config.yaml"` |
| `RepositoryNotFoundError: 401 Client Error` (下载 CALVIN 数据集时) | 未指定 `repo_type='dataset'` | `hf_hub_download` 调用中添加 `repo_type='dataset'` |
| `ModuleNotFoundError: No module named 'libero'` | LIBERO 的 editable install bug | 用 `python setup.py develop` 安装，或设置 `export PYTHONPATH="/workspace/VLA-Adapter/LIBERO:$PYTHONPATH"` |
| `ImportError: cannot import name 'ImageSequenceClip' from 'moviepy.editor'` | moviepy v2 移除了此 API | `pip install "moviepy<2.0"` |
| `numpy` 相关版本冲突 | tensorflow 2.15 要求 numpy<2.0 | `pip install "numpy<2.0.0,>=1.23.5"` |
| `transformers` 版本不对 | sentence-transformers 拉高了版本 | `pip install "transformers==4.40.1" "tokenizers==0.19.1"` |
| 下载 0 个文件（CALVIN 数据集） | 使用了 `snapshot_download` + `allow_patterns`（文件是 zip 不是目录） | 改用 `hf_hub_download` 下载 zip 后 `unzip` |
| `EGL` 渲染相关报错 | 缺少 Mesa/EGL 系统依赖 | `apt-get install -y libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libosmesa6-dev` |

---

## 目录结构

```
/workspace/VLA-Adapter/
├── LIBERO/                              # git clone
├── calvin/                              # git clone --recurse-submodules
│   └── dataset/task_ABC_D/
│       └── validation/                  # 从 HF 下载 zip 解压
│           ├── .hydra/merged_config.yaml  # 必须单独下载
│           ├── lang_annotations/
│           └── episode_*.npz              # ~99000 个文件
├── outputs/                             # HF 下载的 checkpoint
│   ├── LIBERO-{Spatial,Object,Goal,Long}-Pro/
│   └── CALVIN-ABC-Pro/
├── pretrained_models/                   # VLM backbone
├── experiments/robot/libero/
│   └── run_libero_eval.py               # LIBERO 评估入口
├── vla-scripts/
│   └── evaluate_calvin.py               # CALVIN 评估入口
└── evaluation_results/calvin/           # CALVIN 输出（success_rate.txt, result.txt, *.mp4）
```
