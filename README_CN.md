# UniAD 复现笔记

> **UniAD: Planning-oriented Autonomous Driving** (CVPR 2023 Best Paper)  
> 本仓库 fork 自 [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)，在此基础上记录了完整的复现过程、Bug 修复和实验结果。

---

## 复现结果

在 **nuScenes mini** 数据集（val split，81 samples）上使用官方预训练权重评估，结果如下：

### Stage 1：感知阶段（TrackFormer + MapFormer）

| 任务 | 指标 | 本复现 | 论文报告 |
|------|------|--------|----------|
| 3D检测 | mAP | 0.3713 | 0.359 |
| 3D检测 | NDS | 0.3960 | 0.498 |
| 多目标追踪 | AMOTA | 0.486 | 0.359 |
| 多目标追踪 | AMOTP | 1.088 | 1.319 |
| 多目标追踪 | MOTA | 0.465 | — |
| 地图感知 | drivable IoU | 0.621 | — |
| 地图感知 | lanes IoU | 0.221 | — |
| 地图感知 | divider IoU | 0.249 | — |

### Stage 2：端到端完整流水线（全五模块）

| 任务 | 指标 | 本复现 |
|------|------|--------|
| **感知** 3D检测 | mAP | 0.3751 |
| **感知** 3D检测 | NDS | 0.4027 |
| **追踪** | AMOTA | 0.483 |
| **追踪** | AMOTP | 1.076 |
| **追踪** | MOTA | 0.479 |
| **地图** | drivable IoU | 0.695 |
| **地图** | lanes IoU | 0.246 |
| **地图** | divider IoU | 0.301 |
| **占用预测** | IoU (moving) | 67.5 |
| **占用预测** | IoU (static) | 45.3 |
| **占用预测** | PQ (moving/static) | 55.7 / 36.3 |
| **运动预测** | car EPA | 0.612 |
| **运动预测** | pedestrian EPA | 0.353 |
| **规划** | L2 @ 1s | 0.309 m |
| **规划** | L2 @ 2s | 0.896 m |
| **规划** | L2 @ 3s | 1.903 m |
| **规划** | obj 碰撞率 @ 3s | **0.000** |

> 注：论文在完整 nuScenes val 集（6019 samples）上评估，mini 子集结果存在差异属正常现象。

---

## 五模块架构

```
摄像头图像 (6×cameras)
    ↓
BEVFormer (图像特征 → BEV特征)
    ↓
┌─────────────────────────────────────────────┐
│  TrackFormer  →  检测 + 追踪（agent tokens） │
│  MapFormer    →  高精地图语义分割             │
│  MotionFormer →  未来轨迹预测（6s, 12 steps） │
│  OccFormer    →  未来占用流预测               │
│  Planner      →  自车轨迹规划（3s）           │
└─────────────────────────────────────────────┘
```

每个模块的输出作为下一模块的输入，形成端到端联合优化的统一框架。

---

## 环境要求

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 22.04（或 WSL2） |
| Python | 3.9 |
| PyTorch | 2.0.1+cu118 |
| CUDA | 11.8 |
| mmcv-full | 1.6.1 |
| mmdet | 2.26.0 |
| mmdet3d | 1.0.0rc6 |
| mmsegmentation | 0.29.1 |

---

## 快速开始

### 1. 克隆并安装

```bash
git clone https://github.com/YOUR_USERNAME/UniAD.git
cd UniAD
conda create -n uniad2.0 python=3.9 -y
conda activate uniad2.0
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
# 详细安装步骤见 docs/setup_wsl2.md
```

### 2. 准备数据

```bash
# 下载 nuScenes mini 数据集
# 生成数据信息文件
python tools/create_data.py nuscenes --root-path ./data/nuscenes \
    --out-dir ./data/infos --extra-tag nuscenes_infos_temporal \
    --version v1.0-mini --canbus ./data/nuscenes
```

### 3. 下载预训练权重

```bash
mkdir ckpts
# 从 HuggingFace 下载（国内可用 hf-mirror.com）
wget https://hf-mirror.com/OpenDriveLab/UniAD/resolve/main/uniad_base_track_map.pth -P ckpts/
wget https://hf-mirror.com/OpenDriveLab/UniAD/resolve/main/uniad_base_e2e.pth -P ckpts/
```

### 4. 评估

```bash
# Stage 1：感知（检测 + 追踪 + 地图）
./tools/uniad_dist_eval.sh \
    ./projects/configs/stage1_track_map/base_track_map.py \
    ./ckpts/uniad_base_track_map.pth 1

# Stage 2：完整端到端流水线
./tools/uniad_dist_eval.sh \
    ./projects/configs/stage2_e2e/base_e2e.py \
    ./ckpts/uniad_base_e2e.pth 1
```

---

## 相关文档

- [环境搭建指南（WSL2）](docs/setup_wsl2.md) — 从零搭建完整开发环境
- [复现结果详细对比](docs/reproduction_results.md) — 与论文数据的逐指标对比
- [Bug 修复记录](docs/bug_fixes.md) — 10+ 个兼容性 bug 的原因与修复方法

---

## 引用

```bibtex
@inproceedings{hu2023_uniad,
  title={Planning-Oriented Autonomous Driving},
  author={Hu, Yihan and Yang, Jiazhi and Chen, Li and Li, Keyu and Sima, Chonghao and Zhu, Xizhou and Chai, Siqi and Du, Senyao and Lin, Tianwei and Wang, Wenhai and Lu, Lewei and Jia, Xiaosong and Liu, Qiang and Dai, Jifeng and Qiao, Yu and Li, Hongyang},
  booktitle={CVPR},
  year={2023}
}
```
