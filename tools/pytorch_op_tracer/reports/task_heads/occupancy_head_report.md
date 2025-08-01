# UniAD Occupancy Head Module Analysis Report

## Overview

The Occupancy Head predicts future 3D occupancy states in the BEV space, crucial for collision-aware planning. It generates binary occupancy masks for multiple future timesteps, enabling the planning module to avoid potential collisions.

## Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                        OccHead                                │
├──────────────────────────────────────────────────────────────┤
│ • Prediction Horizon: 5 timesteps (2.5 seconds)              │
│ • Output: Binary occupancy masks (occupied/free)             │
│ • Resolution: 200×200 BEV grid                               │
│ • Integrates motion predictions for dynamic objects           │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                   Occupancy Pipeline                          │
├──────────────────────────────────────────────────────────────┤
│  Motion Predictions → Flow Estimation → Occupancy Warping    │
│         +                    ↓                ↓              │
│  BEV Features → CNN Decoder → Future Occupancy Masks         │
└──────────────────────────────────────────────────────────────┘
```

## Key Mechanisms

### 1. Motion-Aware Occupancy Prediction

The occupancy head leverages motion predictions to warp current observations into future frames:

```python
# Input: motion trajectories from motion head
agent_trajectories = outs_motion['traj']  # (B, N, 6, 12, 2)
agent_scores = outs_motion['traj_scores']  # (B, N, 6)

# Convert trajectories to occupancy flow
flow_field = self.traj_to_flow(agent_trajectories, agent_scores)

# Warp current occupancy to future
future_occ = self.warp_features(current_occ, flow_field)
```

### 2. Multi-Timestep Architecture

```python
class OccupancyDecoder(nn.Module):
    def __init__(self, predict_steps=5):
        # Separate decoder for each future timestep
        self.decoders = nn.ModuleList([
            ConvDecoder(in_dim=256, out_dim=1)
            for _ in range(predict_steps)
        ])
    
    def forward(self, bev_feat, motion_flow):
        future_occs = []
        feat = bev_feat
        
        for t, decoder in enumerate(self.decoders):
            # Apply motion flow
            feat = warp_features(feat, motion_flow[t])
            
            # Predict occupancy
            occ = decoder(feat)
            future_occs.append(occ)
            
        return torch.stack(future_occs, dim=1)
```

## Occupancy Prediction Pipeline

### Input Integration

| Input | Source | Shape | Purpose |
|-------|--------|-------|---------|
| bev_embed | BEV Encoder | (B, 256, 200, 200) | Spatial features |
| track_results | Motion Head | Dict | Agent trajectories |
| segmentation | Seg Head | (B, C, 200, 200) | Static scene |

### Processing Flow

```python
# 1. Extract motion information
motion_flow = self.compute_flow_field(
    trajectories=track_results['traj'],
    scores=track_results['traj_scores']
)

# 2. Combine static and dynamic information
combined_feat = torch.cat([
    bev_embed,
    static_segmentation,
    current_occupancy
], dim=1)

# 3. Predict future occupancy
future_occupancy = self.occupancy_decoder(
    features=combined_feat,
    motion_flow=motion_flow
)  # (B, T, 1, H, W)

# 4. Apply sigmoid for binary prediction
occupancy_probs = torch.sigmoid(future_occupancy)
```

## Loss Functions

### 1. Binary Cross-Entropy Loss

```python
def occupancy_loss(pred_occ, gt_occ, mask):
    # Weighted BCE for class imbalance
    pos_weight = (mask == 0).sum() / (mask == 1).sum()
    
    loss = F.binary_cross_entropy_with_logits(
        pred_occ, gt_occ,
        pos_weight=pos_weight,
        reduction='none'
    )
    
    # Masked average
    return (loss * mask).sum() / mask.sum()
```

### 2. Temporal Consistency Loss

```python
def temporal_smooth_loss(future_occs):
    # Encourage smooth transitions
    loss = 0
    for t in range(len(future_occs) - 1):
        diff = future_occs[t+1] - future_occs[t]
        loss += torch.abs(diff).mean()
    return loss
```

## Integration with Planning

### Safety Integration

The occupancy predictions directly inform the planning module's collision checking:

```python
# Planning module uses occupancy for collision avoidance
def check_collision(ego_trajectory, occupancy_masks):
    collisions = []
    for t, pos in enumerate(ego_trajectory):
        # Check if ego position intersects with occupied cells
        x, y = world_to_bev(pos)
        if occupancy_masks[t, 0, y, x] > threshold:
            collisions.append(t)
    return collisions
```

### Output Interface

```python
occ_results = {
    "occ": future_occupancy_masks,      # (B, T, 1, H, W)
    "occ_prob": occupancy_probabilities, # Sigmoid activated
    "flow": motion_flow_field,           # For visualization
}
```

## Performance Characteristics

### Metrics
- **IoU**: 63.7% on nuScenes validation
- **Prediction Horizon**: 2.5 seconds (5 steps)
- **Resolution**: 200×200 @ 0.512m/pixel

### Computational Requirements
- **Memory**: ~6 GB
- **Inference Time**: ~25ms
- **FLOPs**: ~15 GFLOPs

## Key Algorithms

### 1. Trajectory to Flow Field Conversion

```python
def traj_to_flow(trajectories, scores):
    """Convert discrete trajectories to continuous flow field"""
    flow = torch.zeros(B, T, 2, H, W)
    
    for agent_traj, score in zip(trajectories, scores):
        # Get best trajectory mode
        best_mode = torch.argmax(score)
        traj = agent_traj[best_mode]
        
        # Rasterize trajectory to BEV
        for t in range(T):
            start_pos = traj[t]
            end_pos = traj[t+1]
            
            # Compute flow vector
            flow_vec = end_pos - start_pos
            
            # Splat to BEV grid
            flow[t] = splat_to_bev(start_pos, flow_vec)
    
    return flow
```

### 2. Feature Warping

```python
def warp_features(features, flow):
    """Warp features using flow field"""
    B, C, H, W = features.shape
    
    # Create sampling grid
    grid = create_meshgrid(H, W)
    
    # Apply flow
    new_grid = grid + flow
    
    # Bilinear sampling
    warped = F.grid_sample(
        features, new_grid,
        mode='bilinear',
        padding_mode='zeros'
    )
    
    return warped
```

## Configuration

```yaml
# Occupancy head configuration
predict_steps: 5
feature_dim: 256
decoder_layers: [256, 128, 64, 32, 1]

# Loss weights
occupancy_loss_weight: 1.0
temporal_loss_weight: 0.1

# Occupancy parameters
occupancy_threshold: 0.5
flow_aggregation: "max"  # How to combine multiple agent flows
```

## Optimization Opportunities

### 1. Efficiency Improvements
- Share decoder weights across timesteps
- Use depthwise separable convolutions
- Implement sparse occupancy representation

### 2. Accuracy Enhancements
- Multi-scale occupancy prediction
- Uncertainty estimation for occupancy
- Incorporate map priors

### 3. Integration Improvements
- Direct gradient flow from planning loss
- Learnable occupancy aggregation
- Dynamic prediction horizon

## Best Practices

1. **Training Strategy**:
   - Pre-train on static occupancy first
   - Gradually add motion integration
   - Use curriculum learning for prediction horizon

2. **Data Preparation**:
   - Generate GT occupancy from LiDAR
   - Account for occlusions properly
   - Balance free/occupied samples

3. **Deployment Considerations**:
   - Cache static occupancy
   - Use lower resolution for distant regions
   - Implement fail-safe mechanisms