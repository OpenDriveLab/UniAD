# UniAD Planning Head Module Analysis Report

## Overview

The Planning Head represents the culmination of UniAD's hierarchical architecture, responsible for generating safe ego vehicle trajectories based on perception, prediction, and navigation inputs. It demonstrates how end-to-end learning can effectively integrate multiple autonomous driving tasks.

## Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                   PlanningHeadSingleMode                      │
├──────────────────────────────────────────────────────────────┤
│ • Planning Horizon: 6 steps (3 seconds @ 2Hz)                │
│ • Single-mode trajectory generation                          │
│ • Multi-layer safety mechanisms                              │
│ • Navigation command integration (left/straight/right)        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    Feature Integration                        │
├──────────────────────────────────────────────────────────────┤
│  Motion Query (256D) + Track Query (256D) + Nav Embed (256D) │
│                   ↓ MLP Fusion ↓                             │
│              Planning Query (1×256D)                          │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              Attention Module (3 layers)                      │
├──────────────────────────────────────────────────────────────┤
│  • 8-head cross-attention with BEV features                  │
│  • FFN with residual connections                              │
│  • Layer normalization                                        │
└──────────────────────────────────────────────────────────────┘
```

## Input Integration

### Multi-Modal Input Fusion

| Input | Source | Dimension | Purpose |
|-------|--------|-----------|---------|
| `sdc_traj_query` | Motion Head | 256D | Ego motion context |
| `sdc_track_query` | Track Head | 256D | Ego tracking features (detached) |
| `navigation_command` | User/Planner | 3 classes | High-level behavior |
| `bev_embed` | BEV Encoder | 256×200×200 | Spatial context |
| `occupancy_mask` | Occ Head | Binary masks | Collision avoidance |

### Feature Fusion Pipeline

```python
# 1. Navigation embedding
navi_embed = self.navi_embed.weight[command]  # (B, 1, 256)

# 2. Multi-modal concatenation
combined = torch.cat([
    sdc_traj_query,    # Motion features
    sdc_track_query,   # Tracking features (detached)
    navi_embed         # Navigation command
], dim=-1)  # (B, N, 768)

# 3. MLP fusion and aggregation
plan_query = self.mlp_fuser(combined)  # (B, N, 256)
plan_query = plan_query.max(dim=1, keepdim=True)[0]  # (B, 1, 256)
```

## Planning Decision Pipeline

### 1. Spatial Reasoning

The attention module performs cross-attention between planning query and BEV features:

```python
# 3-layer transformer decoder
for layer in self.attn_layers:
    # Cross-attention with BEV spatial features
    plan_query = layer(
        query=plan_query,        # (B, 1, 256)
        key=bev_feat,           # (B, H*W, 256)
        value=bev_feat,         # (B, H*W, 256)
        query_pos=query_pos,    # Positional encoding
        key_pos=bev_pos         # BEV positional encoding
    )
```

### 2. Trajectory Generation

```python
# Regression to trajectory
traj_output = self.reg_branch(plan_query)  # (B, 1, 6, 2)

# Convert to cumulative positions
# Predicts relative displacements, then integrates
ego_fut_trajs = traj_output.cumsum(dim=-2)

# Apply bivariate Gaussian for uncertainty
if self.use_gaussian:
    ego_fut_trajs = bivariate_gaussian_activation(ego_fut_trajs)
```

## Safety Mechanisms

### Multi-Layer Collision Avoidance

#### 1. Training-Time Collision Loss

Three-tier collision checking with different safety margins:

| Margin | Weight | Purpose |
|--------|--------|---------|
| 0.0m | 2.5 | Immediate collision penalty |
| 0.5m | 1.0 | Close proximity warning |
| 1.0m | 0.25 | Safety buffer maintenance |

```python
# Collision loss computation
for delta, weight in [(0.0, 2.5), (0.5, 1.0), (1.0, 0.25)]:
    loss += weight * collision_loss(pred_traj, gt_boxes, delta)
```

#### 2. Inference-Time Optimization

CasADi-based nonlinear optimization for trajectory refinement:

```python
# Optimization formulation
minimize: ||traj - traj_init||² + α * collision_cost
subject to: kinematic_constraints

# Collision cost using Gaussian penalty
collision_cost = Σ exp(-dist²/2σ²) for obstacles in range
```

### Vehicle Model Parameters

```python
# Ego vehicle dimensions
length = 4.084m
width = 1.85m
wheel_base = 2.70m

# Safety parameters
filter_range = 5.0m  # Consider obstacles within 5m
sigma = 1.0m         # Gaussian penalty spread
```

## Loss Functions

### 1. Planning Loss (L2 Distance)

```python
def planning_loss(pred_ego_fut_trajs, gt_ego_fut_trajs, has_ego_fut):
    # L2 distance between predicted and ground truth
    err = torch.sqrt(torch.sum((pred - gt)**2, dim=-1))
    
    # Masked average over valid timesteps
    loss = torch.sum(err * has_ego_fut) / torch.sum(has_ego_fut)
    return loss
```

### 2. Collision Loss

```python
def collision_loss(pred_trajs, gt_boxes, delta=0.0):
    # Expand ego box by safety margin
    ego_box = expand_box(ego_dims, delta)
    
    # Check intersections with future GT boxes
    collisions = box_intersection(ego_box, gt_boxes)
    
    # Binary collision penalty
    return torch.mean(collisions.float())
```

## Performance Metrics

### Key Performance Indicators

| Metric | Target | Description |
|--------|--------|-------------|
| Collision Rate | 0.29% | Percentage of collision events |
| L2 Error (1s) | ~0.5m | Planning accuracy at 1 second |
| L2 Error (2s) | ~1.2m | Planning accuracy at 2 seconds |
| L2 Error (3s) | ~2.0m | Planning accuracy at 3 seconds |

### Computational Efficiency

- **Memory Usage**: ~2 GB (within full model)
- **Inference Time**: <10ms per frame
- **Planning Frequency**: 2 Hz (configurable)

## Integration Best Practices

### 1. Training Strategy

```yaml
# Stage 2 configuration
freeze_bev_encoder: true  # Memory efficiency
planning_loss_weight: 1.0  # Equal task weighting
collision_loss_weight: 5.0  # Emphasize safety
```

### 2. Navigation Command Handling

```python
# Command mapping
NAVIGATION_COMMANDS = {
    0: "TURN_LEFT",
    1: "GOING_STRAIGHT", 
    2: "TURN_RIGHT"
}

# Learnable embeddings for each command
self.navi_embed = nn.Embedding(3, embed_dims)
```

### 3. Gradient Management

```python
# Detach tracking features to prevent gradient interference
sdc_track_query = sdc_track_query.detach()

# This prevents planning gradients from affecting tracking
```

## Optimization Opportunities

### 1. Multi-Modal Planning
- Extend to predict multiple trajectory hypotheses
- Add trajectory scoring mechanism
- Implement behavior-aware mode selection

### 2. Enhanced Safety
- Add predictive collision checking
- Implement comfort constraints
- Include traffic rule compliance

### 3. Computational Optimization
- Sparse attention for efficiency
- Trajectory caching for similar scenarios
- Dynamic planning horizon based on speed

## Key Insights

1. **Hierarchical Integration**: Successfully leverages all upstream modules through attention-based fusion

2. **Safety-First Design**: Multiple layers of collision avoidance ensure safe trajectory generation

3. **Practical Constraints**: Balances planning horizon (3s) with real-time requirements (2Hz)

4. **End-to-End Benefits**: Joint training allows planning objectives to influence feature learning

5. **Navigation Flexibility**: Incorporates high-level commands while maintaining safety

The planning head demonstrates how careful architectural design can integrate complex perception and prediction outputs into safe, executable trajectories for autonomous driving.