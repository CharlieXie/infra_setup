# NTU Cluster Training Runbook

> AI Agent 参考文档。读完即可在 NTU EEE Cluster 02 上提交训练任务。
>
> 集群完整手册：`/projects/chuanlia001ssd/docs/ai-cluster-manual.md`
> 环境搭建详情：`NTU_CLUSTER_EVAL_RUNBOOK.md`（Part 2，第 9–11 节）

---

## 集群关键约束（msc 账户）

| 约束 | 值 |
|------|-----|
| 用户名 | `chuanlia001` |
| 同时最多任务数 | 2 |
| 每种 GPU 最多 | 2 张（V100 为 4） |
| sbatch 最长 | **3 天**（不指定默认 1 小时） |
| srun 最长 | 2 小时，最多 1 GPU（仅调试用） |
| 推荐 GPU | `pro6000`（96GB Blackwell）；V100 不兼容 cu128 |
| Login 节点 | 8GB 内存限制，禁止重计算，断连杀进程 |

---

## 文件结构

```
/home/chuanlia001/envs/openpi/          # conda 环境（eval + training 共用）
/projects/chuanlia001ssd/repos/pi_train/ # 训练代码
    configs/waypoint_joint_libero.yaml   # LIBERO 配置
    configs/waypoint_joint_calvin.yaml   # CALVIN 配置
    scripts/cluster_train_joint.sh       # sbatch 脚本（pro6000:2, 3天）
    scripts/train_waypoint_joint.py      # 训练入口
    logs/                                # 日志
    checkpoints/                         # checkpoint 输出
```

---

## 提交训练

```bash
cd /projects/chuanlia001ssd/repos/pi_train

# LIBERO（默认配置）
sbatch scripts/cluster_train_joint.sh

# CALVIN
sbatch scripts/cluster_train_joint.sh configs/waypoint_joint_calvin.yaml
```

sbatch 脚本已配置：`--gpus=pro6000:2 --constraint=highmem --time=3-00:00:00`

---

## 监控与管理

```bash
squeue -u chuanlia001                        # 查看任务状态
scancel <JOBID>                              # 取消任务
tail -f logs/train_joint-<JOBID>.log         # 实时日志
grep "\[Joint\]" logs/train_joint-<JOBID>.log | tail -5  # 最近 loss
```

---

## 常见问题速查

| 问题 | 解决 |
|------|------|
| `QOSMaxJobsPerUserLimit` | 先 scancel 一个任务 |
| `QOSMaxGRESPerUser` | GPU 配额满，换型号或等待 |
| 多卡 illegal memory access | `pip install "nvidia-nccl-cu12>=2.29"` |
| pip 升级了 torch | `pip install torch==2.7.1 torchvision==0.22.1 --index-url .../cu128` |
