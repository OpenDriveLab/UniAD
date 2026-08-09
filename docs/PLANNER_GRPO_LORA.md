# Planner-only group-relative LoRA post-training

This optional path freezes stage-2 UniAD and updates only planning-head LoRA
parameters plus a learned diagonal Gaussian `log_std`. The original
`PlanningHeadSingleMode` path is unchanged.

The policy samples six 2D displacement increments per rollout and converts
them to waypoints with `cumsum`. Training uses group-normalized open-loop
ADE/FDE/comfort rewards, an exact non-negative KL to the LoRA-disabled SFT
planner, and an auxiliary mean-path ADE loss. Inference always deploys the
deterministic mean path.

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes --canbus ./data/nuscenes \
  --version v1.0-mini --out-dir ./data/infos \
  --extra-tag nuscenes

python tools/train.py \
  projects/configs/planner_posttrain/uniad_planner_grpo_lora.py \
  --work-dir work_dirs/planner_grpo_lora
```

## Preliminary v1.0-mini evidence

Hardware: RTX 4060 Ti 16 GB, batch size 1, FP32. Metrics are open-loop only.

| Measurement | Result |
|---|---:|
| Rollout sample-best minFDE (best logged frame) | 0.052 m |
| Deterministic mean-path ADE (12 frames) | 3.60 m mean / 0.98 m median |
| Deterministic mean-path FDE (12 frames) | 6.83 m mean |
| Mean-path worst case | 19.50 m |

These numbers came from the earlier auxiliary-fit campaign. Its audit found
that the same-forward PPO ratio was always 1.0 and its reparameterized action
made the reported GRPO term ineffective. Therefore they show planner-only
LoRA/auxiliary fitting feasibility, not a causal GRPO gain. This patch removes
that defect by scoring detached actions and tests for non-zero policy-mean
gradients. A fresh full mini comparison is still required.

Sample-best rollout ADE/FDE and deterministic mean-path ADE/FDE are different
metrics and must not be compared as if they were the same estimator. No
trainval planning result or closed-loop collision result is claimed.
