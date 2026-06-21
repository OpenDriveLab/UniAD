# VLM 难例语义归因(端到端失效可视化)

把 UniAD 的逐帧规划评测,延伸到「用开源 VLM 对最难的帧做语义归因」,
并把 VLM 的判断纳入可度量的评测框架——而非只看主观描述。

## 流程
1. 从 `output/planning_per_frame_attr.csv` 按 `l2_3s` 取 worst-15 难例。
2. 每帧喂三路前视相机(FRONT_LEFT/FRONT/FRONT_RIGHT)给 **Qwen2-VL-2B-Instruct**(fp16),
   强制输出 JSON:场景描述 / 是否路口 / 关键障碍 / 为什么难 / 建议 meta-action。
3. **关键设计**:拿 VLM 判的 `is_intersection` 对照 nuScenes 地图真值,
   把"主观解释"变成一个可算召回/一致率的客观实验。
4. 结果 → `output/vlm_attribution.csv` + `vlm_fields.json` → FiftyOne 难例浏览器
   (`tools/fiftyone_vlm_hardcases.py`,数据集 `uniad_vlm_hardcases`)。

## 结果(诚实记录,负结果)
- worst-15 里地图标了 **8 帧路口**,Qwen2-VL-2B 零样本**全部判 False**,
  路口召回 **0%**,且 15 帧 meta-action 全塌成 `keep_lane`。
- 表面 46.7% 的"一致率"只是非路口帧的基率假象,**不可当准确率汇报**。

## 结论与下一步
小 VLM 零样本在难例上发生**输出塌缩 + 任务定义错位**(地图 `is_intersection`
是拓扑标签,未必在前视里可见),实证了:要么上更大模型/微调(Qwen2.5-VL-7B 4bit 待验),
要么改评测设计(多相机输入 / 更贴视觉的定义)。**真正的交付物是"VLM 接入评测闭环 +
可视化归因"这条工程链路**,而非那个 0% 数字本身。
