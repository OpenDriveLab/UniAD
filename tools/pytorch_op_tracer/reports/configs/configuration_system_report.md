# UniAD Configuration System Report

## Overview

UniAD uses a hierarchical configuration system based on MMDetection3D's config framework. The system supports inheritance, variable substitution, and modular composition, enabling flexible experimentation and deployment.

## Configuration Structure

### Directory Layout

```
projects/configs/
├── _base_/
│   ├── datasets/
│   │   └── nus-3d.py          # NuScenes dataset config
│   └── default_runtime.py      # Training runtime settings
├── stage1_track_map/
│   └── base_track_map.py       # Stage 1 perception config
└── stage2_e2e/
    └── base_e2e.py             # Stage 2 end-to-end config
```

## Key Configuration Files

### 1. Stage 1: Perception Foundation

**File**: `stage1_track_map/base_track_map.py`

```python
# Key settings
queue_length = 5  # Temporal frames
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_h_ = 200
bev_w_ = 200
num_query = 900

# Model configuration
model = dict(
    type='UniADTrack',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        frozen_stages=4,
        norm_eval=True,
        dcn=dict(type='DCNv2')
    ),
    # ... task heads
)
```

**Training Strategy**:
- 200 epochs total
- Unfrozen neck and BN
- Focus on tracking (AMOTA: 0.393) and mapping

### 2. Stage 2: End-to-End Integration

**File**: `stage2_e2e/base_e2e.py`

```python
# Inherits from Stage 1
_base_ = ['./stage1_track_map/base_track_map.py']

# Key modifications
queue_length = 3  # Reduced for memory
model = dict(
    type='UniAD',
    freeze_bev_encoder=True,  # Critical for memory
    # Additional task heads
    motion_head=dict(...),
    occ_head=dict(...),
    planning_head=dict(...),
    task_loss_weight=dict(
        track=1.0,
        map=1.0,
        motion=1.0,
        occ=1.0,
        planning=1.0
    )
)
```

## Configuration Components

### Model Architecture

```yaml
# Backbone
img_backbone:
  type: ResNet
  depth: 101
  num_stages: 4
  frozen_stages: 4
  with_cp: False  # Checkpoint for memory saving

# BEV Encoder
pts_bbox_head:
  transformer:
    type: PerceptionTransformer
    embed_dims: 256
    encoder:
      num_layers: 6
      transformerlayers:
        attn_cfgs:
          - type: TemporalSelfAttention
          - type: SpatialCrossAttention
```

### Data Pipeline

#### Training Pipeline
```python
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeCropFlipRotImage'),
    dict(type='GlobalRotScaleTransImage'),
    dict(type='NormalizeMultiviewImage'),
    dict(type='PadMultiViewImage'),
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', ...])
]
```

#### Test Pipeline
```python
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='ResizeCropFlipRotImage'),  # No augmentation
    dict(type='NormalizeMultiviewImage'),
    dict(type='PadMultiViewImage'),
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=['img'])
]
```

### Training Configuration

```python
# Optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01
)

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    min_lr_ratio=1e-3
)

# Training settings
total_epochs = 20  # Stage 2
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1)
```

## Key Configuration Parameters

### Core Parameters

| Parameter | Stage 1 | Stage 2 | Purpose |
|-----------|---------|---------|---------|
| `queue_length` | 5 | 3 | Temporal context frames |
| `num_query` | 900 | 900 | Object detection queries |
| `freeze_bev_encoder` | False | True | Memory optimization |
| `total_epochs` | 200 | 20 | Training duration |
| `batch_size` | 1 | 1 | Per GPU |

### BEV Grid Configuration

```python
# Spatial coverage
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# 102.4m × 102.4m × 8m volume

# BEV resolution
bev_h, bev_w = 200, 200
# 0.512m per pixel

# Voxel size
voxel_size = [0.2, 0.2, 8]
```

### Loss Weights

```python
# Stage 1
loss_weights = {
    'cls': 2.0,
    'bbox': 0.25,
    'iou': 0.0,    # Disabled
    'heatmap': 1.0
}

# Stage 2 - Balanced multi-task
task_loss_weight = {
    'track': 1.0,
    'map': 1.0,
    'motion': 1.0,
    'occ': 1.0,
    'planning': 1.0
}
```

## Dataset Configuration

### NuScenes Settings

```python
dataset_type = 'NuScenesTrackDataset'
data_root = 'data/nuscenes/'
ann_file_train = 'data/infos_train_10sweeps_withvelo_filter_True.pkl'
ann_file_val = 'data/infos_val_10sweeps_withvelo_filter_True.pkl'

# Class names (10 classes)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Input modality
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True  # CAN bus
)
```

## Runtime Configuration

### Training Runtime

```python
# Logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# Distributed training
dist_params = dict(backend='nccl')
find_unused_parameters = True

# Mixed precision
fp16 = dict(loss_scale=512.)
```

### Evaluation Settings

```python
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    metric=['bbox', 'track', 'map']
)
```

## Configuration Best Practices

### 1. Extending Configurations

```python
# Inherit from base config
_base_ = ['./base_config.py']

# Override specific settings
model = dict(
    pts_bbox_head=dict(
        num_query=1200  # Increase queries
    )
)
```

### 2. Variable Substitution

```python
# Define variables
_dim_ = 256
_num_levels_ = 4

# Use throughout config
model = dict(
    embed_dims=_dim_,
    num_feature_levels=_num_levels_
)
```

### 3. Conditional Configuration

```python
# Stage-specific settings
if stage == 1:
    queue_length = 5
    freeze_bev_encoder = False
else:
    queue_length = 3
    freeze_bev_encoder = True
```

## Common Modifications

### 1. Memory Optimization
```python
# Reduce queue length
queue_length = 3  # From 5

# Enable gradient checkpointing
img_backbone = dict(with_cp=True)

# Reduce batch size accumulation
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
```

### 2. Training Acceleration
```python
# Increase learning rate with larger batch
optimizer = dict(lr=4e-4)  # From 2e-4

# Reduce validation frequency
evaluation = dict(interval=5)  # From 1
```

### 3. Task-Specific Tuning
```python
# Adjust task weights
task_loss_weight = dict(
    track=1.0,
    map=0.5,     # Reduce map weight
    motion=2.0,  # Emphasize motion
    occ=1.0,
    planning=1.5
)
```

## Configuration Loading

The configuration system supports:
- **Inheritance**: `_base_` imports
- **Composition**: Merge multiple configs
- **Override**: Command-line arguments
- **Validation**: Automatic type checking

Example usage:
```bash
# Load and modify config
python tools/train.py configs/base.py \
    --cfg-options model.num_query=1000 \
                  optimizer.lr=1e-4
```

This flexible configuration system enables rapid experimentation while maintaining reproducibility and clarity in the complex UniAD architecture.