# OpenPI Waypoint VLA — LIBERO 评测 Runbook

> **目标读者**: AI Agent。适用场景：已有训练好的 VLM + AE checkpoint，在新机器上从零搭建评测环境。
>
> **最后验证**: 2026-02-25，Ubuntu 24.04，CUDA 12.8
>
> **耗时预估**: 环境搭建 ~10 分钟，模型加载 ~2 分钟，30 episodes 评测 ~15-20 分钟

---

## Agent 行为准则

1. **`sleep` 最多 30 秒**，循环轮询长任务。
2. **后台任务用 `&`**，输出重定向文件，用 `tail` 检查。
3. **发现错误立刻读日志**，不要盲目重试。
4. **stdin 交互用管道传入**：`echo "N" | command`，避免挂起。
5. **写文件前确认目录可写**（rclone 下载目录属主 `nobody:nogroup`，无写权限）。

---

## 快速 Checklist

```
□ touch ~/.no_auto_tmux
□ sudo apt-get install -y ffmpeg pkg-config build-essential libosmesa6-dev libgles2 libegl1
□ cd /workspace && git clone openpi (pytorch_lora_blackwell)
□ git submodule update --init --recursive
□ uv sync（后台）→ 完成后装 TF、打 transformers patch
□ 生成 dataset_statistics.json（约 60s，见第 4 节）
□ 安装评测依赖（robosuite、libero）
□ 修改 eval config → 填入 checkpoint 路径
□ 启动评测 → 查看结果
```

---

## 1. 基础环境

```bash
touch ~/.no_auto_tmux

git config --global user.email "chuanliang.xie@gmail.com"
git config --global user.name "chuanliang"

# 构建依赖 + MuJoCo OSMesa 渲染（评测必须，缺少会报 glGetError 错误）
sudo apt-get install -y ffmpeg pkg-config build-essential \
    libosmesa6-dev libgles2 libegl1
```

---

## 2. 克隆代码仓库

```bash
cd /workspace
git clone https://<PAT>@github.com/CharlieXie/openpi.git
cd openpi
git checkout pytorch_lora_blackwell
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

验证：
```bash
cd /workspace/openpi && git branch  # * pytorch_lora_blackwell
```

---

## 3. 配置 Python 环境

### 3.1 uv sync（后台运行）

```bash
cd /workspace/openpi
# 确认 av override 已存在（pytorch_lora_blackwell 分支通常已预置）
grep "override-dependencies" pyproject.toml  # 应含 av>=13.1.0,<14.0.0

GIT_LFS_SKIP_SMUDGE=1 uv sync > /tmp/uv_sync.log 2>&1 &
echo "PID=$!"
```

监控（实测约 3-10 分钟）：
```bash
sleep 30 && tail -5 /tmp/uv_sync.log
# 完成标志: "Installed N packages"
```

### 3.2 安装 TensorFlow（uv sync 完成后）

```bash
cd /workspace/openpi
uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"
# ⚠️ 必须 2.15.0，>=2.16 与 ml_dtypes 冲突
```

验证：
```bash
.venv/bin/python -c "import tensorflow as tf; print('TF:', tf.__version__)"
# TF: 2.15.0
```

### 3.3 应用 transformers 补丁

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

## 4. 生成归一化统计量

> 评测时 `NormalizationHelper` 需要 `action` 和 `proprio` 统计量做反归一化。

**如果已有 `/workspace/data/dataset_statistics.json`，跳过本节。**

否则从 LIBERO RLDS 数据计算（需先有数据，见注释）：

```bash
cd /workspace/openpi

# ⚠️ 目标路径必须是 /workspace/data/（可写），不要写到 RLDS 数据目录（nobody:nogroup 无写权限）
.venv/bin/python - << 'PYEOF' > /tmp/compute_stats.log 2>&1 &
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import numpy as np, json

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
# 期望: (66984, 7), (66984, 8)
def stats(arr):
    return {'mean': arr.mean(0).tolist(), 'std': arr.std(0).tolist(),
            'q01': np.percentile(arr, 1, 0).tolist(), 'q99': np.percentile(arr, 99, 0).tolist(),
            'min': arr.min(0).tolist(), 'max': arr.max(0).tolist()}
out = {'libero_object_no_noops': {'action': stats(all_actions), 'proprio': stats(all_proprios), 'num_samples': len(all_actions)}}
path = '/workspace/data/dataset_statistics.json'
with open(path, 'w') as f: json.dump(out, f, indent=2)
print('Saved to', path)
PYEOF
echo "stats PID=$!"
```

监控（约 60 秒）：
```bash
sleep 30 && tail -5 /tmp/compute_stats.log
# 完成标志: "Saved to /workspace/data/dataset_statistics.json"
```

验证：
```bash
.venv/bin/python -c "
import json; d = json.load(open('/workspace/data/dataset_statistics.json'))
k = list(d.keys())[0]
print('action dims:', len(d[k]['action']['q99']))   # 7
print('proprio dims:', len(d[k]['proprio']['q99'])) # 8
"
```

---

## 5. 安装评测依赖

```bash
cd /workspace/openpi

uv pip install --python .venv/bin/python \
    robosuite==1.4.1 transforms3d bddl easydict "gym==0.26.2"

uv pip install --python .venv/bin/python -e third_party/libero

# 修复 PyTorch 2.6+ weights_only 兼容性
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
    third_party/libero/libero/libero/benchmark/__init__.py
```

验证（⚠️ 用管道传 "N"，避免交互提示挂起）：
```bash
echo "N" | PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH \
.venv/bin/python -c "
from libero.libero import benchmark
bm = benchmark.get_benchmark_dict()['libero_object']()
print(f'LIBERO Object: {bm.n_tasks} tasks')  # 10 tasks
"
```

---

## 6. 配置评测参数

编辑 `configs/eval_waypoint_libero.yaml`：

```yaml
# checkpoint 路径两种格式均可（eval_libero.py 自动拼接 model.safetensors）：
#   格式 A（训练 checkpoint 有 step 子目录）: /workspace/openpi/checkpoints/waypoint_vlm_libero/5000
#   格式 B（直接提供模型目录）:              /workspace/models/vlm
vlm_checkpoint: /workspace/models/vlm
ae_checkpoint: /workspace/models/ae

dataset_statistics_path: /workspace/data/dataset_statistics.json

num_trials_per_task: 3   # 每 task 3 次，共 30 episodes
num_steps_wait: 10
video_out_path: data/libero/videos_wp
```

启动前核查：
```bash
ls /workspace/models/vlm/model.safetensors   && echo "✓ VLM"
ls /workspace/models/ae/model.safetensors    && echo "✓ AE"
ls /workspace/data/dataset_statistics.json   && echo "✓ stats"
.venv/bin/python -c "from OpenGL.GL import glGetError; print('✓ OpenGL')"
```

---

## 7. 启动评测

```bash
cd /workspace/openpi
mkdir -p logs data/libero/videos_wp

MUJOCO_GL=osmesa \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH \
PYTHONFAULTHANDLER=1 \
.venv/bin/python -u -m openpi.waypoint.eval_libero \
    --config configs/eval_waypoint_libero.yaml \
    2>&1 | tee logs/eval_libero.log
```

监控（另开终端或后台运行后 tail）：
```bash
tail -f logs/eval_libero.log | grep -E "Task|SUCCESS|FAIL|Overall"
```

**正常启动标志**（约 2 分钟后出现）：
```
INFO:__main__:VLM model init: 20s
INFO:__main__:VLM weight load: 2s
INFO:__main__:AE model init: 15s
INFO:__main__:Task 0/10: pick_up_the_alphabet_soup...
INFO:__main__:  [replan 1] VLM: 7 waypoints, ...
```

---

## 8. 查看结果

评测结束后日志输出：
```
Overall success rate: XX.XX% (N/30)
  pick_up_the_alphabet_soup_and_place_it_in_the_basket: XX.XX% (n/3)
  ...
```

视频文件：
```bash
ls data/libero/videos_wp/
# rollout_{task_name}_t{trial}_{success|failure}.mp4
```

**GPU 内存参考**：VLM (float32, ~11.7 GB) + AE (bfloat16, ~7.5 GB) ≈ 19.2 GB，单张 RTX 4090 (24 GB) 可运行。

---

## 已知问题

### 问题 1：`glGetError` AttributeError（MuJoCo OSMesa）

```
AttributeError: 'NoneType' object has no attribute 'glGetError'
```

**修复**：`sudo apt-get install -y libosmesa6-dev libgles2 libegl1`（已在第 1 节）

验证：`.venv/bin/python -c "from OpenGL.GL import glGetError; print('OK')"`

---

### 问题 2：LIBERO 验证命令挂起

**原因**：benchmark 初始化时弹出 `Y/N` 交互提示。

**修复**：`echo "N" | .venv/bin/python -c "..."` 管道传入。

---

### 问题 3：VLM checkpoint key 不匹配

```
RuntimeError: Cannot find PaliGemma weights in checkpoint
```

**原因**：checkpoint 保存时 key 前缀为 `paligemma_with_expert.paligemma.*`。

**修复**：`eval_libero.py` 的 `load_vlm()` 已自动 remap，无需手动处理。

---

### 问题 4：`torch.load weights_only` 错误

```
_pickle.UnpicklingError: Weights only load failed
```

**修复**：
```bash
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
    third_party/libero/libero/libero/benchmark/__init__.py
```
（已在第 5 节执行）

---

## 关键路径

| 资源 | 路径 |
|------|------|
| VLM checkpoint | `/workspace/models/vlm/` |
| AE checkpoint | `/workspace/models/ae/` |
| Dataset statistics | `/workspace/data/dataset_statistics.json` |
| Eval config | `/workspace/openpi/configs/eval_waypoint_libero.yaml` |
| Eval log | `/workspace/openpi/logs/eval_libero.log` |
| 视频输出 | `/workspace/openpi/data/libero/videos_wp/` |
| LIBERO RLDS | `/workspace/data/libero/libero_object_no_noops/libero_object_no_noops/1.0.0/` |
