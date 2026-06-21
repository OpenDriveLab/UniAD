# UniAD Planner 输入审计 — 有没有「ego-status 开环泄漏」?

## 背景
开环 L2 评测的著名坑(AD-MLP / BEV-Planner《Is Ego Status All You Need?》):
不少端到端方法把 ego 速度/加速度/yaw-rate 直接喂进 planner,导致开环 L2 虚高——
planner 靠外推自车运动学就能压低 L2,并非真在感知环境。本节审计 UniAD 是否有同样捷径。

## 方法
grep planner 代码 + 通读 `planning_head.py` 的 forward。

## 发现:plan_query 只由三样东西拼成(planning_head.py:168, fuser_dim=3)
| 输入 | 来源 | 关键处理 |
|---|---|---|
| `sdc_traj_query` | motion_head 预测的 SDC 轨迹 query | motion_head.py:152,201,取最后一层 |
| `sdc_track_query` | tracking 的 SDC track embedding | `.detach()`(planning_head.py:160)梯度不回传感知 |
| `navi_embed` | 离散驾驶命令 command∈{0,1,2} | `nn.Embedding(3, d)`(planning_head.py:51,166) |

```
command(0/1/2) ─► navi_embed ─┐
SDC traj query ───────────────┤─ cat → mlp_fuser → max over modes → [1,1,256]
SDC track query(.detach) ─────┘ │
+ pos_embed → cross-attn × 3 → BEV
→ reg MLP → cumsum 积分成轨迹
```

## 关键结论:planner 没有显式 ego-status 捷径
- 搜不到 `ego_lcf_feat / ego_status / ego_his_traj` 喂给 planner;速度/加速度/yaw 没直接拼进 plan_query。
- `use_can_bus=True` 只在 BEVFormer 编码器(base_e2e:156)做 ego-motion 时序补偿/BEV 对齐,**不是 planner 输入**。
- 配置里的 `sdc_planning / sdc_planning_mask / gt_sdc_*` 是 GT 监督标签,走 data pipeline 的 collect,**不是前向输入**。

## 含义 → 对 UniAD 该审的开环坑是「command 先验」而非「ego-status」
UniAD 不像被批的那类方法有裸 ego-status 捷径;ego 信息只经 ①离散 command ②自预测 SDC 轨迹
③SDC track + can_bus **隐式**进入。所以对 UniAD,开环最该量化的是 **command 先验依赖**:
L2 有多少来自"跟着 3 类离散命令走"的先验,而非真感知。

## 下一步实验(待跑):command-prior ablation
推理时把 `navi_embed` 置零/随机化,重跑 eval,对比 L2(1/2/3s)虚高量。
若 L2 几乎不变 → planner 重度依赖命令先验 → 开环指标偏乐观(方法论结论,低算力)。
