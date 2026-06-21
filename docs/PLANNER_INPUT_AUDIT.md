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

## 实验结果:command-prior ablation(已跑,v1.0-mini,69 valid frames)
推理时把 `navi_embed` 置零(`ABLATE_COMMAND=1` 环境开关,默认关、形状不变、逻辑不动;
一键两遍脚本 `tools/run_command_ablation.sh`),其余完全一致,重跑 eval 对比逐帧 L2:

| metric | baseline(command ON) | no_command(zeroed) | Δ(n−b) | 相对 |
|---|---|---|---|---|
| l2_1s | 0.3245 | 0.3568 | +0.0323 | +10% |
| l2_2s | 1.0146 | 1.2349 | +0.2203 | +22% |
| l2_3s | 2.2338 | 2.6933 | +0.4595 | +21% |

paired mean&#124;Δl2_3s&#124; = **0.5183**(69 帧逐帧配对,证明是真改变,非均值抵消的假象)。

### 读法(实测 Δ 大,而非 Δ≈0)
- 去掉离散命令后 L2 显著变差,且**随时域放大**(1s 几乎不动 → 2/3s 大幅恶化)——
  机理一致:3 类命令(左/右/直行)消歧的是**路由**,而路由只在长时域(2–3s 的转向)才分叉。
- 结论:**navi command 是 planner 的承重输入(load-bearing),不是可有可无的标签。**
  UniAD 长时域开环 L2 里有相当一块(**~0.46m / 21% @3s**)是"离散命令喂出来的"而非纯感知——
  这把 AD-MLP/BEV-Planner 对开环指标的批评**在 UniAD 上量化了**:指标被高层命令先验抬着。
- 诚实边界:本消融只证明命令**必要**(去掉就垮),**没**证明 planner 退化/忽略感知
  (未做 command-only / 随机命令 / zero-BEV 对照)。要坐实"开环偏乐观"还需这些控制实验,留作后续。

> 注:数字在 v1.0-mini(69 valid frames)上、绝对值噪声大;关键是 **Δ 的方向与随时域增大的趋势**,而非绝对 L2。
> 复现:`bash tools/run_command_ablation.sh`(`ABLATE_COMMAND` 开关见 `planning_head.py` 的 `navi_embed` 处)。
