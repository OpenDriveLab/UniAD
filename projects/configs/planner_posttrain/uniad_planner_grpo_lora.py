_base_ = ['../stage2_e2e/base_e2e.py']

# The info files below must be generated from nuScenes v1.0-mini when
# reproducing the preliminary experiment in docs/PLANNER_GRPO_LORA.md.
data = dict(samples_per_gpu=1, workers_per_gpu=2)

model = dict(
    planner_posttrain_only=True,
    planning_head=dict(
        type='PlanningHeadGRPO',
        loss_collision=[],
        use_col_optim=False,
        grpo=dict(
            num_samples=6,
            kl_weight=0.01,
            aux_weight=1.0,
            lora_rank=8,
            lora_alpha=16.0,
            init_log_std=-1.5,
            ade_weight=1.0,
            fde_weight=1.0,
            comfort_weight=0.1,
        ),
    ),
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.0)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
load_from = 'ckpts/uniad_base_e2e.pth'
total_epochs = 3
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
find_unused_parameters = True
