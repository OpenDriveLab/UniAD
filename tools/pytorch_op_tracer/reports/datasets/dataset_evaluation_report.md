# UniAD Dataset and Evaluation System Report

## Overview

UniAD uses the nuScenes dataset with custom extensions for temporal tracking, future trajectory labels, and occupancy annotations. The evaluation system provides comprehensive metrics across all tasks.

## Dataset Structure

### NuScenes Extensions

UniAD extends the standard nuScenes dataset with:

1. **Temporal Tracking**: Instance IDs across frames
2. **Future Trajectories**: 6-second future paths for agents
3. **Occupancy Labels**: Future occupancy states
4. **Planning Ground Truth**: Expert driving trajectories

### Data Organization

```
data/
├── nuscenes/
│   ├── maps/                    # HD maps
│   ├── samples/                 # Camera images
│   ├── sweeps/                  # Additional sensor data
│   └── v1.0-trainval/          # Annotations
├── infos_train_10sweeps_withvelo_filter_True.pkl
├── infos_val_10sweeps_withvelo_filter_True.pkl
└── others/
    ├── motion_anchor_infos_mode6.pkl
    ├── sdc_occ_gt_train.pkl
    └── sdc_occ_gt_val.pkl
```

## Data Pipeline

### Training Pipeline Components

```python
train_pipeline = [
    # 1. Data Loading
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D', 
         with_bbox_3d=True,
         with_label_3d=True,
         with_instance_id=True,
         with_future_traj=True),
    
    # 2. Augmentation
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeCropFlipRotImage',
         data_aug_conf={
             'resize_lim': (0.47, 0.625),
             'final_dim': (320, 800),
             'bot_pct_lim': (0.0, 0.0),
             'rot_lim': (0.0, 0.0),
             'H': 900, 'W': 1600,
             'rand_flip': True
         }),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         scale_ratio_range=[0.95, 1.05]),
         
    # 3. Normalization
    dict(type='NormalizeMultiviewImage',
         mean=[103.530, 116.280, 123.675],
         std=[1.0, 1.0, 1.0]),
         
    # 4. Formatting
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=[
        'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds',
        'gt_fut_traj', 'gt_fut_traj_mask', 'gt_past_traj',
        'gt_past_traj_mask', 'gt_sdc_bbox', 'gt_sdc_label',
        'gt_sdc_fut_traj', 'gt_sdc_fut_traj_mask',
        'sdc_planning', 'sdc_planning_mask', 'command',
        'map_gt_bboxes_3d', 'map_gt_labels_3d'
    ])
]
```

### Data Augmentation Strategy

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Resize | 0.47-0.625x | Scale variation |
| Random Flip | Horizontal | Left-right symmetry |
| Rotation | ±22.5° | Viewpoint variation |
| Color Jitter | Standard | Lighting robustness |
| Global Transform | ±22.5°, 0.95-1.05x | Scene-level augmentation |

## Task-Specific Data

### 1. Tracking Data

```python
# Instance tracking annotations
instance_inds = [...]  # Instance IDs across frames
gt_past_traj = [...]   # Past 4 timesteps @ 0.5s
gt_fut_traj = [...]    # Future 4 timesteps @ 0.5s

# Tracking specific info
track_info = {
    'timestamp': float,
    'scene_token': str,
    'prev_bboxes': list,
    'prev_exists': list
}
```

### 2. Motion Prediction Data

```python
# Agent trajectories
motion_labels = {
    'gt_agent_boxes': Tensor(N, 10),     # Current 3D boxes
    'gt_agent_feats': Tensor(N, 8, 20),  # Past features
    'gt_agent_fut_traj': Tensor(N, 6, 2), # 6-second future
    'gt_agent_fut_mask': Tensor(N, 6)     # Valid mask
}

# Pre-computed anchors
anchor_info = load('motion_anchor_infos_mode6.pkl')
# Contains 6 trajectory modes per agent class
```

### 3. Occupancy Data

```python
# Future occupancy states
occ_labels = {
    'gt_segmentation': List[Tensor],     # Current segmentation
    'gt_instance': List[Tensor],         # Instance masks
    'gt_flow': Tensor(2, H, W),         # Optical flow
    'gt_backward_flow': Tensor(2, H, W), # Backward flow
    'gt_occ_img': Tensor(T, H, W)       # Future occupancy
}
```

### 4. Planning Data

```python
# Ego vehicle planning
planning_labels = {
    'sdc_planning': Tensor(6, 2),        # 3-second trajectory
    'sdc_planning_mask': Tensor(6),      # Valid steps
    'command': int,                      # 0: left, 1: straight, 2: right
    'gt_future_boxes': List[Tensor]      # Future object states
}
```

## Evaluation Metrics

### 1. Detection & Tracking (AMOTA)

```python
# Average Multi-Object Tracking Accuracy
tracking_metrics = {
    'mAP': mean_average_precision,
    'mATE': mean_translation_error,
    'mASE': mean_scale_error,
    'mAOE': mean_orientation_error,
    'mAVE': mean_velocity_error,
    'mAAE': mean_attribute_error,
    'AMOTA': average_multi_object_tracking_accuracy,
    'AMOTP': average_multi_object_tracking_precision,
    'IDS': identity_switches
}

# Key metric: AMOTA
AMOTA = (1 - (FN + FP + IDS) / GT) * (1/T) * sum(recall)
```

### 2. Motion Prediction

```python
motion_metrics = {
    'minADE': minimum_average_displacement_error,
    'minFDE': minimum_final_displacement_error,
    'MR': miss_rate,
    'EPA': end_point_accuracy
}

# Computed over 6 trajectory modes
minADE = min(ADE across modes)
minFDE = min(FDE across modes)
MR = fraction(minFDE > 2.0m)
```

### 3. Segmentation (Map)

```python
seg_metrics = {
    'mIoU': mean_intersection_over_union,
    'lane_iou': lane_specific_iou,
    'vehicle_iou': vehicle_segmentation_iou,
    'ped_cross_iou': pedestrian_crossing_iou
}
```

### 4. Occupancy Prediction

```python
occ_metrics = {
    'IoU': intersection_over_union,
    'VPQ': video_panoptic_quality,
    'soft-IoU': soft_intersection_over_union
}

# Evaluated at multiple thresholds
for t in [0.5, 1.0, 1.5, 2.0, 2.5]:  # seconds
    compute_iou(pred_occ[t], gt_occ[t])
```

### 5. Planning

```python
planning_metrics = {
    'L2': l2_distance_error,
    'collision_rate': collision_percentage,
    avg_L2: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # Per timestep
    avg_col: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
}

# Collision checking
def check_collision(ego_traj, future_objects):
    ego_box = expand_dims(ego_vehicle_box)
    for t, pos in enumerate(ego_traj):
        if box_collision(ego_box, future_objects[t]):
            return True
    return False
```

## Evaluation Pipeline

### 1. Data Preparation

```python
# Generate predictions
results = model.simple_test(data_loader)

# Format for evaluation
formatted_results = format_results(
    results,
    classes=class_names,
    coord_type='lidar'
)
```

### 2. Multi-Task Evaluation

```python
# Evaluate all tasks
metrics = {}

# Detection & Tracking
metrics['track'] = eval_tracking(
    formatted_results['track'],
    gt_annos,
    metric='AMOTA'
)

# Motion Prediction
metrics['motion'] = eval_motion(
    formatted_results['motion'],
    gt_trajectories,
    metric=['minADE', 'minFDE', 'MR']
)

# Planning
metrics['planning'] = eval_planning(
    formatted_results['planning'],
    gt_planning,
    check_collision=True
)
```

### 3. Visualization

```python
# Visualize results
visualizer = UniADVisualizer()
visualizer.show_result(
    img=camera_images,
    bev_bbox=track_results,
    bev_seg=seg_results,
    trajectories=motion_results,
    ego_traj=planning_results,
    occ_map=occ_results
)
```

## Dataset Statistics

### NuScenes Overview

| Aspect | Training | Validation |
|--------|----------|------------|
| Scenes | 700 | 150 |
| Frames | 28,130 | 6,019 |
| Annotations | 1.4M | 300K |
| Classes | 10 | 10 |
| Cameras | 6 | 6 |

### Temporal Statistics

- **Sequence Length**: 20 seconds @ 2Hz
- **History**: 2 seconds (4 frames)
- **Future**: 6 seconds (12 frames)
- **Total Context**: 8 seconds

## Data Loading Optimization

### 1. Caching Strategy

```python
# Cache frequently accessed data
cache_config = dict(
    cache_mode='disk',
    cache_dir='data/cache/',
    cache_scenes=True,
    cache_annotations=True
)
```

### 2. Multi-Processing

```python
data_loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True
)
```

### 3. Memory Management

```python
# Load only necessary frames
queue_dataset = VideoDataset(
    queue_length=3,  # Only load 3 frames
    skip_ratio=1,    # No frame skipping
    cache_prev_data=True
)
```

## Custom Extensions

### 1. Future Trajectory Generation

```python
def generate_future_labels(current_box, velocity, T=12):
    """Generate future trajectory GT from current state"""
    future_traj = []
    for t in range(T):
        # Constant velocity model
        future_pos = current_pos + velocity * (t + 1) * 0.5
        future_traj.append(future_pos)
    return torch.stack(future_traj)
```

### 2. Occupancy Label Generation

```python
def generate_occupancy_labels(lidar_points, future_boxes):
    """Convert future boxes to occupancy grid"""
    occ_grid = torch.zeros(T, H, W)
    for t in range(T):
        # Rasterize boxes to BEV
        occ_grid[t] = rasterize_boxes(future_boxes[t])
    return occ_grid
```

## Best Practices

### 1. Data Quality
- Filter static scenes for motion tasks
- Ensure temporal consistency in tracking
- Validate planning trajectories for feasibility

### 2. Evaluation Protocol
- Use official nuScenes evaluation kit
- Report metrics on full validation set
- Include per-class breakdowns

### 3. Debugging
- Visualize augmented samples
- Check data loading speed
- Monitor GPU memory during loading

The comprehensive dataset and evaluation system enables thorough assessment of UniAD's multi-task performance while maintaining compatibility with standard benchmarks.