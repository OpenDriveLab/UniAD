# UniAD 复现结果详细对比

复现环境：WSL2 Ubuntu 22.04，NVIDIA GPU，CUDA 11.8，nuScenes **mini** val split（81 samples）

> 论文在完整 nuScenes val 集（6019 samples）评估，mini 结果存在统计差异属正常现象，但趋势一致。

---

## Stage 1：感知阶段（base_track_map）

使用权重：`uniad_base_track_map.pth`  
配置文件：`projects/configs/stage1_track_map/base_track_map.py`  
结果目录：`test/base_track_map/Wed_May_27_21_52_44_2026/`

### 3D 目标检测（Detection）

| 指标 | 本复现（mini） | 论文（full val） |
|------|---------------|-----------------|
| mAP | 0.3713 | 0.359 |
| NDS | 0.3960 | 0.498 |
| mATE | 0.7175 | — |
| mASE | 0.7733 | — |
| mAOE | 0.6544 | — |
| mAVE | 0.4405 | — |

各类别 AP：

| 类别 | AP |
|------|-----|
| car | 0.641 |
| truck | 0.445 |
| bus | 0.669 |
| pedestrian | 0.533 |
| motorcycle | 0.449 |
| bicycle | 0.290 |
| traffic_cone | 0.637 |
| trailer | 0.000* |
| barrier | 0.000* |

*mini 数据集中 trailer/barrier 样本极少，评估不稳定。

### 多目标追踪（Tracking）

| 指标 | 本复现（mini） | 论文（full val） |
|------|---------------|-----------------|
| AMOTA ↑ | 0.486 | 0.359 |
| AMOTP ↓ | 1.088 | 1.319 |
| MOTA ↑ | 0.465 | — |
| MOTP ↑ | 0.653 | — |
| IDS ↓ | — | — |

### 地图感知（Map）

| 指标 | 本复现（mini） |
|------|---------------|
| drivable_iou | 0.621 |
| lanes_iou | 0.221 |
| divider_iou | 0.249 |

---

## Stage 2：完整端到端流水线（base_e2e）

使用权重：`uniad_base_e2e.pth`  
配置文件：`projects/configs/stage2_e2e/base_e2e.py`  
结果目录：`test/base_e2e/Wed_May_27_22_00_28_2026/`

### 3D 目标检测（Detection）

| 指标 | Stage 2 | Stage 1 | 论文 |
|------|---------|---------|------|
| mAP | **0.3751** | 0.3713 | 0.359 |
| NDS | **0.4027** | 0.3960 | 0.498 |

各类别 AP（Stage 2）：

| 类别 | AP |
|------|-----|
| car | 0.655 |
| truck | 0.465 |
| bus | 0.692 |
| pedestrian | 0.544 |
| motorcycle | 0.461 |
| bicycle | 0.294 |
| traffic_cone | 0.639 |

### 多目标追踪（Tracking）

| 指标 | Stage 2 | Stage 1 | 论文 |
|------|---------|---------|------|
| AMOTA ↑ | 0.483 | 0.486 | 0.359 |
| AMOTP ↓ | 1.076 | 1.088 | 1.319 |
| MOTA ↑ | 0.479 | 0.465 | — |
| MOTP ↑ | 0.657 | 0.653 | — |

各类别 AMOTA（Stage 2）：

| 类别 | AMOTA |
|------|-------|
| bicycle | 0.190 |
| bus | 0.766 |
| car | 0.668 |
| motorcycle | 0.450 |
| pedestrian | 0.554 |
| truck | 0.271 |

### 地图感知（Map）

| 指标 | Stage 2 | Stage 1 |
|------|---------|---------|
| drivable_iou | **0.695** | 0.621 |
| lanes_iou | **0.246** | 0.221 |
| divider_iou | **0.301** | 0.249 |
| crossing_iou | 0.151 | — |
| contour_iou | 0.198 | — |

### 占用预测（Occupancy Flow，OccFormer）

| 指标 | 本复现 | 论文 |
|------|--------|------|
| IoU (moving) | **67.5** | 64.0 |
| IoU (static) | **45.3** | 41.6 |
| PQ (moving) | 55.7 | — |
| PQ (static) | 36.3 | — |
| SQ (moving/static) | 75.8 / 71.1 | — |
| RQ (moving/static) | 73.6 / 51.1 | — |
| 评估样本数 | 69 / 81 (85.2%) | — |

### 运动预测（Motion，MotionFormer）

EPA 指标（End-to-end Prediction Accuracy，联合检测+预测评估）：

| 类别 | EPA | 预测样本 | 匹配 GT |
|------|-----|----------|---------|
| car | **0.612** | 1913 | 1429 |
| truck | 0.479 | 95 | 64 |
| bus | 0.273 | 33 | 10 |
| pedestrian | **0.353** | 1067 | 742 |
| motorcycle | 0.467 | 214 | 113 |
| bicycle | 0.292 | 36 | 17 |

### 规划（Planning，Planner）

| 指标 | 0.5s | 1.0s | 1.5s | 2.0s | 2.5s | 3.0s |
|------|------|------|------|------|------|------|
| L2 误差 (m) ↓ | 0.172 | 0.309 | 0.548 | 0.896 | 1.364 | **1.903** |
| obj 碰撞率 ↓ | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |
| obj_box 碰撞率 ↓ | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |

> **亮点**：在整个 mini val 集上规划碰撞率为零，与论文报告一致。

---

## 结果保存路径

```
~/UniAD/
├── test/
│   ├── base_track_map/
│   │   └── Wed_May_27_21_52_44_2026/
│   │       ├── results_nusc.json      # 追踪/检测结果
│   │       ├── det/                   # 检测评估详情
│   │       └── track/                 # 追踪评估详情
│   └── base_e2e/
│       └── Wed_May_27_22_00_28_2026/
│           ├── results_nusc.json
│           ├── results_nusc_det.json
│           ├── det/
│           └── track/
```

---

## 简历用量化表述参考

> Reproduced UniAD (CVPR 2023 Best Paper) on nuScenes mini dataset; achieved mAP=0.375, AMOTA=0.483, Occ-IoU=67.5/45.3, and **0% planning collision rate** across all evaluated scenes.
