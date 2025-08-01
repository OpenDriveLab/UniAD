# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

UniAD is a Planning-oriented Autonomous Driving framework that unifies perception, prediction, and planning tasks. Instead of standalone modular design, UniAD casts tasks hierarchically following a planning-oriented philosophy.

## Core Architecture

### Main Components

1. **Core Detector Classes**:
   - `UniADTrack` (projects/mmdet3d_plugin/uniad/detectors/uniad_track.py): Base perception module for tracking and mapping
   - `UniAD` (projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py): Full end-to-end model integrating all tasks

2. **Task Heads** (projects/mmdet3d_plugin/uniad/dense_heads/):
   - `BEVFormerTrackHead` (track_head.py): 3D object tracking
   - `PansegformerHead` (panseg_head.py): BEV segmentation/mapping
   - `MotionHead` (motion_head.py): Motion prediction
   - `OccHead` (occ_head.py): Occupancy prediction
   - `PlanningHeadSingleMode` (planning_head.py): Trajectory planning

3. **Configuration Structure**:
   - Stage 1: projects/configs/stage1_track_map/ (perception only)
   - Stage 2: projects/configs/stage2_e2e/ (full model)

### Training Pipeline

UniAD is trained in two stages:
- **Stage 1**: Perception modules (track + map), queue_length=5
- **Stage 2**: All modules (freeze BEV encoder), queue_length=3

## Development Commands

### Training
```bash
# Stage 1 training (perception)
./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/base_track_map.py 8

# Stage 2 training (end-to-end)
./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/base_e2e.py 8
```

### Evaluation
```bash
# Evaluate stage 1 model
./tools/uniad_dist_eval.sh ./projects/configs/stage1_track_map/base_track_map.py ./ckpts/uniad_base_track_map.pth 8

# Evaluate stage 2 model
./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/base_e2e.py ./ckpts/uniad_base_e2e.pth 8
```

### Visualization
```bash
python ./tools/analysis_tools/visualize/run.py \
    --predroot /PATH/TO/RESULTS.pkl \
    --out_folder /PATH/TO/OUTPUT \
    --demo_video test_demo.avi \
    --project_to_cam True
```

### Testing a Single Configuration
```bash
# Test with specific GPU
CUDA_VISIBLE_DEVICES=0 python tools/test.py [config_file] [checkpoint] --eval bbox

# Distributed testing
python -m torch.distributed.launch --nproc_per_node=8 tools/test.py [config_file] [checkpoint] --launcher pytorch --eval bbox
```

## Key Technical Details

### Model Configuration
- **BEV Grid**: 200x200, resolution 0.512m per pixel
- **Point Cloud Range**: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
- **Feature Dimension**: 256
- **Temporal Aggregation**: 3-5 frames depending on stage

### Task Loss Weights (Stage 2)
- track: 1.0
- map: 1.0
- motion: 1.0
- occ: 1.0
- planning: 1.0

### Performance Targets
- **Stage 1**: AMOTA ~0.390 for tracking
- **Stage 2**: 
  - Motion: ~0.705 minADE
  - Occupancy: ~63.7% IoU
  - Planning: ~0.29% avg collision rate

## Important Notes

1. **GPU Memory Requirements**:
   - Stage 1: ~50GB (can reduce to ~30GB with queue_length=3)
   - Stage 2: ~17GB (BEV encoder frozen)

2. **Pretrained Weights**:
   - BEVFormer backbone: bevformer_r101_dcn_24ep.pth
   - Stage 1: uniad_base_track_map.pth
   - Stage 2: uniad_base_e2e.pth

3. **Dataset**: Uses nuScenes dataset with custom preprocessing for occupancy flow and planning labels

4. **Dependencies**: Requires specific versions - torch==1.9.1, mmcv-full==1.4.0, mmdet==2.14.0, mmdet3d==0.17.1