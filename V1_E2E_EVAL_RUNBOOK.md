# OpenPI v1_e2e Joint Model — LIBERO 评测 Runbook

> **目标读者**: AI Agent。适用场景：已有训练好的 Joint（VLM+AE 共享 backbone）checkpoint，在新机器上从零搭建评测环境。
>
> **最后验证**: 2026-03-19，Ubuntu 24.04，CUDA 12.8，NVIDIA RTX PRO 6000 Blackwell（~95 GB 显存）
>
> **耗时预估**: 环境搭建 ~15 分钟，模型加载 ~2 分钟，100 episodes 评测 ~40-60 分钟

---

## Agent 行为准则

1. **轮询长任务用循环**，不要固定 `sleep 30`；模式：`while ! grep -q "完成标志" log; do sleep 10 && tail -2 log; done`。
2. **后台任务用 `&`**，输出重定向文件，用 `tail` 检查。
3. **发现错误立刻读日志**，不要盲目重试。
4. **stdin 交互用管道传入**：`echo "N" | command`，避免挂起。
5. **写文件前确认目录可写**（`/workspace/data/` owner 为 nobody，无写权限，写到 `/workspace/openpi/data/` 或 `/workspace/data/libero/`）。

---

## 快速 Checklist

```
□ touch ~/.no_auto_tmux
□ sudo apt-get install -y ffmpeg pkg-config build-essential libosmesa6-dev libgles2 libegl1
□ git config global user
□ cd /workspace && git clone openpi（v1_e2e 分支）
□ GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
□ uv sync（后台）→ 循环轮询 → 完成标志: "Installed N packages"
□ 安装 TF 2.15.0 + tensorflow-datasets 4.9.3（循环轮询）
□ 打 transformers patch → 验证 "patch OK"
□ 确认 dataset_statistics.json 路径（/workspace/data/dataset_statistics.json）
□ 安装评测依赖（robosuite 循环轮询、libero）
□ 修复 torch.load weights_only
□ mkdir -p .torch_cache（torch_compile 编译缓存）
□ 启动 tmux 评测（含 TORCHINDUCTOR_CACHE_DIR）→ tail logs/eval_libero.log
```

---

## 1. 基础环境

```bash
touch ~/.no_auto_tmux

git config --global user.email "chuanliang.xie@gmail.com"
git config --global user.name "chuanliang"

# MuJoCo OSMesa 渲染依赖（缺少会报 glGetError 错误）
sudo apt-get install -y ffmpeg pkg-config build-essential \
    libosmesa6-dev libgles2 libegl1
```

---

## 2. 克隆代码仓库

```bash
cd /workspace
GIT_LFS_SKIP_SMUDGE=1 git clone https://<PAT>@github.com/CharlieXie/openpi.git
cd openpi
git fetch origin v1_e2e
git checkout v1_e2e
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

验证：
```bash
cd /workspace/openpi && git branch  # * v1_e2e
```

> **注意**：v1_e2e 分支是 Joint VLM+AE 共享 backbone 版本，与 pytorch_lora_blackwell（分离 VLM+AE）不同，使用单一 checkpoint。

---

## 3. 配置 Python 环境

### 3.1 uv sync（后台运行）

```bash
cd /workspace/openpi
grep "override-dependencies" pyproject.toml  # 应含 av>=13.1.0,<14.0.0

GIT_LFS_SKIP_SMUDGE=1 uv sync > /tmp/uv_sync.log 2>&1 &
echo "PID=$!"
```

监控（实测约 3-10 分钟）：
```bash
# 循环轮询，完成即退出，避免固定 sleep 浪费时间
while ! grep -q "Installed" /tmp/uv_sync.log 2>/dev/null; do
    sleep 10 && tail -2 /tmp/uv_sync.log
done
echo "uv sync done"
# 完成标志: "Installed N packages"
```

> **加速提示（重复部署）**：如果同一台机器需要多次搭建环境（例如重建容器），可以把 `.venv/` 目录打包备份，下次直接解压，跳过 uv sync（节省 3-10 分钟）：
> ```bash
> # 备份（首次完成后）
> tar -czf /workspace/venv_backup.tar.gz -C /workspace/openpi .venv
> # 还原（新机器/重建后）
> tar -xzf /workspace/venv_backup.tar.gz -C /workspace/openpi
> ```

### 3.2 安装 TensorFlow（uv sync 完成后）

```bash
cd /workspace/openpi
uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3" \
    > /tmp/tf_install.log 2>&1 &
echo "PID=$!"
# 循环轮询，完成即退出
while ! grep -q "Installed" /tmp/tf_install.log 2>/dev/null; do
    sleep 10 && tail -2 /tmp/tf_install.log
done
echo "TF install done"

.venv/bin/python -c "import tensorflow as tf; print('TF:', tf.__version__)"
# TF: 2.15.0
```

### 3.3 升级 NCCL（Blackwell 多卡必须）

```bash
cd /workspace/openpi
uv pip install --python .venv/bin/python "nvidia-nccl-cu12>=2.29"
# PyTorch 自带的 nvidia-nccl-cu12==2.26.2 在 Blackwell (sm_120) 多卡通信时有 illegal memory access bug
```

### 3.4 应用 transformers 补丁

```bash
cd /workspace/openpi
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/

.venv/bin/python -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print('transformers patch OK')
"
```

---

## 4. 确认 dataset_statistics.json

**如果已有 `/workspace/data/dataset_statistics.json`，跳过本节。**

本次发现统计文件已存在于 `/workspace/data/libero/dataset_statistics.json`，直接复制到期望路径：

```bash
# 检查
ls /workspace/data/dataset_statistics.json 2>/dev/null && echo "EXISTS" || echo "NOT FOUND"
ls /workspace/data/libero/dataset_statistics.json 2>/dev/null && echo "libero/ EXISTS"

# 若只有 libero/ 下有，则复制
cp /workspace/data/libero/dataset_statistics.json /workspace/data/dataset_statistics.json

# 验证维度
.venv/bin/python -c "
import json; d = json.load(open('/workspace/data/dataset_statistics.json'))
k = list(d.keys())[0]
print('action dims:', len(d[k]['action']['q99']))   # 7
print('proprio dims:', len(d[k]['proprio']['q99'])) # 8
"
```

> 若两处都不存在，参考 `OPENPI_EVAL_RUNBOOK.md` 第 4 节从 RLDS 数据重新计算（约 60s）。

---

## 5. 安装评测依赖

```bash
cd /workspace/openpi

# robosuite 及辅助库
uv pip install --python .venv/bin/python \
    robosuite==1.4.1 transforms3d bddl easydict "gym==0.26.2" \
    > /tmp/robosuite_install.log 2>&1 &
echo "PID=$!"
# 循环轮询，完成即退出
while ! grep -q "Installed" /tmp/robosuite_install.log 2>/dev/null; do
    sleep 10 && tail -2 /tmp/robosuite_install.log
done
echo "robosuite install done"

# LIBERO（editable 安装）
uv pip install --python .venv/bin/python -e third_party/libero

# 修复 PyTorch 2.6+ weights_only 兼容性
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
    third_party/libero/libero/libero/benchmark/__init__.py
```

验证（⚠️ 管道传 "N"，避免交互提示挂起）：
```bash
echo "N" | PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH \
.venv/bin/python -c "
from libero.libero import benchmark
bm = benchmark.get_benchmark_dict()['libero_object']()
print(f'LIBERO Object: {bm.n_tasks} tasks')  # 10 tasks
"
```

验证 OpenGL（OSMesa）：
```bash
.venv/bin/python -c "from OpenGL.GL import glGetError; print('✓ OpenGL')"
```

---

## 6. 评测配置

使用 `configs/eval_waypoint_joint_libero.yaml`（v1_e2e 分支专用，joint model 配置）：

```yaml
robot_type: libero
task_suite: libero_object

# Joint model 单一 checkpoint（同时含 VLM + AE 权重）
joint_checkpoint: /workspace/models/e2e

dataset_statistics_path: /workspace/data/dataset_statistics.json

# 模型维度
model_action_dim: 32
model_proprio_dim: 32
horizon_steps: 32
num_waypoints: 7
vlm_max_token_len: 256
ae_max_token_len: 64

# 模型架构
paligemma_variant: gemma_2b
action_expert_variant: gemma_300m
precision: bfloat16

norm_type: q99
torch_compile: true

# 评测
num_trials_per_task: 10   # 10 task × 10 trials = 100 episodes
num_steps_wait: 10

# 视频输出
video_out_path: data/libero/videos_wp_joint_object
```

启动前核查：
```bash
ls /workspace/models/e2e/model.safetensors    && echo "✓ Joint model"
ls /workspace/data/dataset_statistics.json    && echo "✓ stats"
.venv/bin/python -c "from OpenGL.GL import glGetError; print('✓ OpenGL')"
```

---

## 7. 启动评测（tmux）

```bash
cd /workspace/openpi
mkdir -p logs data/libero/videos_wp_joint_object .torch_cache

# 杀掉旧 session（如有），启动新 session
tmux kill-session -t eval 2>/dev/null; sleep 1
tmux new-session -d -s eval \
    'cd /workspace/openpi && MUJOCO_GL=osmesa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_CACHE_DIR=/workspace/openpi/.torch_cache PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH PYTHONFAULTHANDLER=1 .venv/bin/python -u -m openpi.waypoint.eval_libero --config configs/eval_waypoint_joint_libero.yaml 2>&1 | tee logs/eval_libero.log'
echo "tmux session eval started"
```

> **`TORCHINDUCTOR_CACHE_DIR` 说明**：`torch_compile: true` 会在首次运行时编译 CUDA kernel（耗时 5-10 分钟），编译结果缓存在 `.torch_cache/`。第二次及之后运行直接复用缓存，模型加载后即可正常速度推理。如果更换了模型结构或 PyTorch 版本，需清空缓存：`rm -rf /workspace/openpi/.torch_cache`。

监控：
```bash
# 实时过滤关键信息
tail -f logs/eval_libero.log | grep -E "Task|SUCCESS|FAIL|Overall|replan"

# 或直接 attach（Ctrl+B, D 退出不杀进程）
tmux attach -t eval
```

**正常启动标志**（约 2 分钟后出现）：
```
INFO:__main__:Device: cuda:0
INFO:__main__:Task 0/10: pick_up_the_alphabet_soup...
INFO:__main__:  [replan 1] VLM: 7 waypoints, 7 valid, durations=[...], vlm_time=...ms
INFO:__main__:    ae[0]: shape=(32, 32), execute=..., ae_time=~160ms
```

---

## 8. 查看结果

评测结束后日志输出：
```
Overall success rate: XX.XX% (N/100)
  pick_up_the_alphabet_soup_and_place_it_in_the_basket: XX.XX% (n/10)
  ...
```

视频文件：
```bash
ls data/libero/videos_wp_joint_object/
# rollout_{task_name}_t{trial}_{success|failure}.mp4
```

**GPU 内存参考**：Joint model (bfloat16) ≈ 12-15 GB，RTX PRO 6000 Blackwell (95 GB) 极宽裕。

---

## 关键路径

| 资源 | 路径 |
|------|------|
| Joint checkpoint | `/workspace/models/e2e/model.safetensors` |
| Dataset statistics | `/workspace/data/dataset_statistics.json` |
| Eval config | `/workspace/openpi/configs/eval_waypoint_joint_libero.yaml` |
| Eval log | `/workspace/openpi/logs/eval_libero.log` |
| 视频输出 | `/workspace/openpi/data/libero/videos_wp_joint_object/` |
| LIBERO RLDS | `/workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/` |

---

## 与旧版 Runbook 的差异（vs OPENPI_EVAL_RUNBOOK.md）

| 项目 | 旧版（pytorch_lora_blackwell） | 新版（v1_e2e） |
|------|-------------------------------|----------------|
| 分支 | `pytorch_lora_blackwell` | `v1_e2e` |
| 模型结构 | 独立 VLM + 独立 AE（两个 checkpoint） | Joint 共享 backbone（单一 checkpoint） |
| 配置文件 | `eval_waypoint_libero.yaml` | `eval_waypoint_joint_libero.yaml` |
| Checkpoint 路径参数 | `vlm_checkpoint` + `ae_checkpoint` | `joint_checkpoint` |
| Config key | `max_token_len` | `vlm_max_token_len` + `ae_max_token_len` |
| 显存占用 | ~19.2 GB（VLM 11.7 + AE 7.5） | ~12-15 GB（joint bfloat16） |
