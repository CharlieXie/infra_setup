# OpenPI Waypoint VLA — Agent Runbook 索引

> **最后验证**: 2026-02-25，硬件: 2× RTX PRO 6000 Blackwell (97.9 GB)，Ubuntu 24.04，CUDA 12.8

本 Runbook 拆分为两个专项文档，按需取用：

| 文档 | 适用场景 | 预计耗时 |
|------|---------|---------|
| **[OPENPI_EVAL_RUNBOOK.md](./OPENPI_EVAL_RUNBOOK.md)** | 已有训练好的 checkpoint，在新机器上跑 LIBERO 评测 | 环境 ~10 min，评测 ~20 min |
| **[OPENPI_TRAINING_RUNBOOK.md](./OPENPI_TRAINING_RUNBOOK.md)** | 从零开始训练 AE + VLM | clone→step=0 约 ~15 min |

与设计规范文档配合使用：[OPENPI_WAYPOINT_VLA_SETUP.md](./OPENPI_WAYPOINT_VLA_SETUP.md)

---

## Agent 行为准则（所有操作均适用）

1. **`sleep` 最多 30 秒**，循环轮询长任务（不用 `block_until_ms > 30000`）。
2. **后台任务用 `&`**，输出重定向文件，用 `tail` 检查。
3. **tmux send-keys 每次只发一条命令**，间隔 `sleep 2`，避免 bash 解析混乱。
4. **发现错误立刻读日志**，不要盲目重试，先 `tail -50 <logfile>` 定位根因。
5. **路径严格按照文档**，不要自行发明。
6. **需要 stdin 输入的命令用管道提前传入**（如 `echo "N" | command`），避免挂起。
7. **写文件前先确认目标目录可写**（rclone/gsutil 下载目录属主可能为 `nobody:nogroup`）。

---

## 共用环境搭建（两个场景都需要）

```bash
# 1. 禁用 vast.ai auto-tmux
touch ~/.no_auto_tmux

# 2. Git
git config --global user.email "chuanliang.xie@gmail.com"
git config --global user.name "chuanliang"

# 3. 系统依赖（构建 + MuJoCo OSMesa 渲染）
sudo apt-get install -y ffmpeg pkg-config build-essential \
    libosmesa6-dev libgles2 libegl1

# 4. 克隆 openpi
cd /workspace
git clone https://<PAT>@github.com/CharlieXie/openpi.git
cd openpi && git checkout pytorch_lora_blackwell
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive

# 5. uv sync（后台）
GIT_LFS_SKIP_SMUDGE=1 uv sync > /tmp/uv_sync.log 2>&1 &

# 6. 安装 TF + transformers patch（uv sync 完成后）
uv pip install --python .venv/bin/python "tensorflow==2.15.0" "tensorflow-datasets==4.9.3"
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```
