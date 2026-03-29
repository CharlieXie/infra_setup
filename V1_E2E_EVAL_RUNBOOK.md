# V1 E2E 评测环境 Runbook

> 目标读者：AI Agent。已有训练好的 Joint（VLM+AE 共享 backbone）checkpoint，在新机器上从零搭建评测环境。
>
> 最后验证：2026-03-29，Ubuntu 24.04，CUDA 12.8，RTX PRO 6000 Blackwell (~95 GB)
>
> 耗时预估：环境搭建 ~15 分钟，模型加载 ~2 分钟，100 episodes 评测 ~40-60 分钟

---

## Agent 行为准则

1. **轮询长任务用循环**：`while ! grep -q "完成标志" log; do sleep 10 && tail -2 log; done`。
2. **后台任务用 `&`**，输出重定向文件，用 `tail` 检查。
3. **发现错误立刻读日志**，不要盲目重试。
4. **stdin 交互用管道传入**：`echo "N" | command`，避免挂起。

---

## 1. 基础环境

```bash
touch ~/.no_auto_tmux

git config --global user.email "chuanliang.xie@gmail.com"
git config --global user.name "chuanliang"

sudo apt-get install -y ffmpeg pkg-config build-essential libosmesa6-dev libgles2 libegl1
```

---

## 2. 克隆代码

```bash
cd /workspace
GIT_LFS_SKIP_SMUDGE=1 git clone https://<PAT>@github.com/CharlieXie/openpi.git
cd openpi
git fetch origin v1.2_e2e && git checkout v1.2_e2e
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
git branch  # 验证: * v1.2_e2e
```

---

## 3. Python 环境

```bash
cd /workspace/openpi

# 3.1 uv sync（后台，3-10 分钟）
GIT_LFS_SKIP_SMUDGE=1 uv sync > /tmp/uv_sync.log 2>&1 &
while ! grep -q "Installed" /tmp/uv_sync.log 2>/dev/null; do sleep 10 && tail -2 /tmp/uv_sync.log; done
echo "uv sync done"

# 3.2 TF（后台，1-2 分钟）
uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3" > /tmp/tf.log 2>&1 &
while ! grep -q "Installed" /tmp/tf.log 2>/dev/null; do sleep 10 && tail -2 /tmp/tf.log; done
.venv/bin/python -c "import tensorflow as tf; print('TF:', tf.__version__)"  # 2.15.0

# 3.3 升级 NCCL（PyTorch 自带 2.26.2 在 Blackwell 多卡有 bug）
uv pip install --python .venv/bin/python "nvidia-nccl-cu12>=2.29"

# 3.4 transformers patch
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
.venv/bin/python -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print('patch OK')
"
```

---

## 4. 生成 dataset_statistics.json

**已有统计文件且 action=7 维、proprio=6 维则跳过。**

```bash
cd /workspace/openpi
.venv/bin/python scripts/compute_wp_norm_stats.py \
    --rlds_dir <RLDS_DIR> \
    --robot_type <libero|calvin> \
    --output_dir <OUTPUT_DIR>
```

LIBERO 示例（约 60s）：
```bash
.venv/bin/python scripts/compute_wp_norm_stats.py \
    --rlds_dir /workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0 \
    --robot_type libero \
    --output_dir /workspace/data
```

验证：
```bash
.venv/bin/python -c "
import json, sys; d = json.load(open(sys.argv[1]))
print('action:', len(d['action']['q99']), 'proprio:', len(d['proprio']['q99']), 'steps:', d['num_transitions'])
" <path_to_dataset_statistics.json>
# 期望: action: 7  proprio: 6
```

---

## 5. 安装评测依赖

```bash
cd /workspace/openpi

# robosuite（后台，1-2 分钟）
uv pip install --python .venv/bin/python \
    robosuite==1.4.1 transforms3d bddl easydict "gym==0.26.2" > /tmp/robo.log 2>&1 &
while ! grep -q "Installed" /tmp/robo.log 2>/dev/null; do sleep 10 && tail -2 /tmp/robo.log; done
echo "robosuite done"

# LIBERO
uv pip install --python .venv/bin/python -e third_party/libero

# 修复 PyTorch 2.6+ weights_only
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
    third_party/libero/libero/libero/benchmark/__init__.py
```

验证：
```bash
echo "N" | PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH \
.venv/bin/python -c "
from libero.libero import benchmark
bm = benchmark.get_benchmark_dict()['libero_object']()
print(f'LIBERO Object: {bm.n_tasks} tasks')
"
# 期望: 10 tasks

.venv/bin/python -c "from OpenGL.GL import glGetError; print('OpenGL OK')"
```

---

## 6. Checkpoint 准备

eval 自动优先加载 `model_merged.safetensors`，不存在则加载 `model.safetensors`。`joint_checkpoint` 指向包含该文件的目录。

| 训练方式 | checkpoint 内容 | 评测前操作 |
|----------|-----------------|-----------|
| 全量微调（`lora_enabled: false`） | `model.safetensors` | 无需处理 |
| LoRA 微调（`lora_enabled: true`） | `lora.safetensors` | **必须先 merge** ↓ |

**LoRA merge**（全量微调跳过）：
```bash
cd /workspace/openpi
.venv/bin/python scripts/merge_lora.py \
    --base <base_model.safetensors> \
    --lora <checkpoint_dir>/<step>/lora.safetensors \
    --config <training_config.yaml> \
    --output <checkpoint_dir>/<step>/model_merged.safetensors
```

---

## 7. 启动前核查

```bash
ls <joint_checkpoint_dir>/model_merged.safetensors 2>/dev/null \
    || ls <joint_checkpoint_dir>/model.safetensors   && echo "✓ model"
ls <dataset_statistics_path>                         && echo "✓ stats"
.venv/bin/python -c "from OpenGL.GL import glGetError; print('✓ OpenGL')"
```

---

## 8. 启动评测（tmux）

评测配置文件：`configs/eval_waypoint_joint_libero.yaml`，按需修改 `joint_checkpoint`、`dataset_statistics_path`、`task_suite`、`num_trials_per_task`、`video_out_path`。

```bash
cd /workspace/openpi
mkdir -p logs .torch_cache

tmux kill-session -t eval 2>/dev/null; sleep 1
tmux new-session -d -s eval \
    'cd /workspace/openpi && \
     MUJOCO_GL=osmesa \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     TORCHINDUCTOR_CACHE_DIR=/workspace/openpi/.torch_cache \
     PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH \
     PYTHONFAULTHANDLER=1 \
     .venv/bin/python -u -m openpi.waypoint.eval_libero \
         --config configs/eval_waypoint_joint_libero.yaml \
         2>&1 | tee logs/eval_libero.log'
echo "tmux session eval started"
```

> `torch_compile: true` 首次编译 CUDA kernel 需 5-10 分钟，缓存在 `.torch_cache/`。后续运行直接复用。

---

## 9. 监控与结果

```bash
tail -f logs/eval_libero.log | grep -E "Task|SUCCESS|FAIL|Overall|replan"
tmux attach -t eval   # Ctrl+B, D 退出不杀进程
```

**正常启动标志**（约 2 分钟后）：
```
INFO:__main__:Task 0/10: pick_up_the_alphabet_soup...
INFO:__main__:  [replan 1] VLM: 7 waypoints, 7 valid, durations=[...], vlm_time=...ms
```

**评测完成标志**：
```
Overall success rate: XX.XX% (N/100)
```

视频：`data/libero/videos_*/rollout_{task}_t{trial}_{success|failure}.mp4`
