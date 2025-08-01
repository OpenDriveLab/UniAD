# UniAD Motion Head Module Analysis Report

## Overview

The Motion Head is responsible for multi-agent trajectory prediction in UniAD, forecasting future paths for all detected vehicles. It leverages track head outputs and BEV features to predict multiple trajectory hypotheses per agent.

## Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                        MotionHead                             │
├──────────────────────────────────────────────────────────────┤
│ • Prediction Horizon: 12 steps (6 seconds @ 0.5s)            │
│ • Trajectory Modes: 6 hypotheses per agent                   │
│ • Anchor-based approach with learnable embeddings            │
│ • Multi-level coordinate systems (agent/ego/scene)           │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              MotionTransformerDecoder (3 layers)              │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Intention Interaction (self-attention)             │
│  Layer 2: Track-Agent Interaction (cross-attention)          │
│  Layer 3: Map Interaction + BEV Deformable Attention         │
└──────────────────────────────────────────────────────────────┘
```

## Key Mechanisms

### 1. Anchor-Based Trajectory Prediction

The system uses pre-computed trajectory anchors for efficient multi-modal prediction:

```python
# Three levels of anchor embeddings
anchor_embeddings = {
    'agent_level': (K, 6, 12, 2),      # Local agent coordinates
    'scene_level_ego': (K, 6, 12, 2),   # Ego-centric coordinates
    'scene_level_offset': (K, 6, 12, 2) # Relative offsets
}

# K = number of agent classes
# 6 = trajectory modes
# 12 = prediction steps
# 2 = (x, y) coordinates
```

### 2. Multi-Agent Processing Pipeline

```
Track Outputs → Agent Filtering → Query Construction → Motion Transformer
     ↓               ↓                    ↓                    ↓
Track Queries   Vehicle IDs      Combined Queries      Trajectory Predictions
Track Boxes     Filter List      + SDC Query          (B, A, 6, 12, 2)
SDC Embedding
```

### 3. Hierarchical Attention Mechanism

#### Layer 1: Intention Interaction
```python
# Self-attention among trajectory anchors
# Models interactions between different trajectory modes
intention_query = trajectory_query.flatten(1, 2)  # (B, A*P, D)
intention_feat = self_attention(intention_query)
```

#### Layer 2: Track-Agent Interaction
```python
# Cross-attention between agents
# Models inter-agent dependencies
agent_query = trajectory_query.mean(dim=2)  # (B, A, D)
interaction_feat = cross_attention(agent_query, track_query)
```

#### Layer 3: Map & BEV Interaction
```python
# Deformable attention with BEV features
# Incorporates spatial context
reference_points = predicted_trajectories
bev_feat = deformable_attention(query, bev_embed, reference_points)
```

## Motion Prediction Pipeline

### Input Processing

| Input | Source | Shape | Purpose |
|-------|--------|-------|---------|
| track_query | Track Head | (B, A, 256) | Agent representations |
| track_bbox | Track Head | (B, A, 10) | 3D bounding boxes |
| bev_embed | BEV Encoder | (B, 256, H, W) | Spatial features |
| sdc_embedding | Track Head | (B, 1, 256) | Ego vehicle query |

### Trajectory Generation Process

```python
# 1. Initialize trajectory queries
traj_query = anchor_embed.weight[agent_classes]  # (B, A, 6, 256)

# 2. Add positional encodings
traj_query += level_embed + class_embed + agent_embed

# 3. Transform through decoder layers
for layer in decoder_layers:
    traj_query = layer(traj_query, track_query, bev_embed)
    
# 4. Predict trajectory offsets
traj_reg = regression_branch(traj_query)  # (B, A, 6, 12, 2)

# 5. Apply cumulative sum for smooth trajectories
predicted_trajectories = reference_points + traj_reg.cumsum(dim=-2)
```

## Loss Functions

### Multi-Component Loss Design

| Component | Type | Weight | Description |
|-----------|------|--------|-------------|
| Classification | CrossEntropy | 1.0 | Mode selection |
| Regression | L1 Loss | 1.0 | Trajectory coordinates |
| ADE | L2 Distance | - | Average displacement error |
| FDE | L2 Distance | - | Final displacement error |
| Miss Rate | Binary | - | Prediction accuracy |

### Loss Computation
```python
def loss_single(self, traj_preds, traj_scores, gt_trajs, gt_modes):
    # 1. Mode selection via Hungarian matching
    matched_indices = self.matcher(traj_preds, gt_trajs)
    
    # 2. Classification loss for mode prediction
    loss_cls = F.cross_entropy(traj_scores, gt_modes)
    
    # 3. Regression loss for matched trajectories
    loss_reg = F.l1_loss(traj_preds[matched], gt_trajs)
    
    # 4. Metrics computation
    min_ade = compute_ade(best_trajectory, gt_trajectory)
    min_fde = compute_fde(best_trajectory[-1], gt_trajectory[-1])
    
    return loss_cls + loss_reg
```

## Performance Characteristics

### Memory Usage
- **Trajectory Queries**: A × 6 × 256 × 4 bytes
- **Anchor Embeddings**: K × 6 × 12 × 2 × 4 bytes
- **Attention Maps**: A² × 4 bytes per layer
- **Total per Frame**: ~4 GB (with 100 agents)

### Computational Complexity
- **Self-Attention**: O(A²P²) for A agents, P modes
- **Cross-Attention**: O(A²D) for feature dimension D
- **Deformable Attention**: O(APK) for K sampling points
- **Total**: O(A²P² + A²D + APK)

## Key Algorithms

### 1. Trajectory Anchor Mining
```python
# Pre-compute common trajectory patterns from training data
# Cluster trajectories by agent type and motion pattern
# Store as learnable embeddings for fast inference
```

### 2. Multi-Coordinate System Transformation
```python
# Agent → Ego coordinates
ego_coords = agent_coords @ rotation_matrix + translation

# Ego → Scene coordinates  
scene_coords = ego_coords + ego_position

# Maintain consistency across transformations
```

### 3. Nonlinear Trajectory Optimization
```python
# Optional kinematic constraint enforcement
if use_nonlinear_optimizer:
    # Apply vehicle dynamics constraints
    # Ensure trajectory smoothness
    # Respect maximum acceleration/deceleration
    optimized_traj = motion_smoother(raw_traj, vehicle_params)
```

## Configuration

### Key Parameters
```yaml
# Prediction settings
predict_steps: 12
predict_modes: 6
use_nonlinear_optimizer: true

# Model architecture
num_decoder_layers: 3
decoder_hidden_dim: 256
num_heads: 8

# Loss weights
cls_weight: 1.0
reg_weight: 1.0

# Agent filtering
vehicle_id_list: [0, 1, 2, 3, 4, 6, 7]  # Vehicle classes only
```

## Integration with Other Modules

### Dependencies
- **Track Head**: Provides agent queries and bounding boxes
- **BEV Encoder**: Supplies spatial context features
- **Map Head**: Lane and road structure information

### Output Interface
```python
motion_results = {
    "traj": predicted_trajectories,      # (B, A, 6, 12, 2)
    "traj_scores": mode_scores,          # (B, A, 6)
    "track_query": updated_track_query,  # For downstream heads
    "sdc_traj": ego_trajectory,          # For planning head
    "sdc_traj_scores": ego_scores        # Ego trajectory confidence
}
```

## Optimization Opportunities

### 1. Efficiency Improvements
- Sparse attention for distant agents
- Trajectory caching for static agents
- Parallel mode prediction

### 2. Accuracy Enhancements
- Scene-specific anchor learning
- Social pooling for dense traffic
- Map-aware trajectory constraints

### 3. Real-time Optimization
- Model quantization
- Reduced prediction horizon for highway
- Dynamic agent filtering

## Best Practices

1. **Training Strategy**:
   - Pre-train anchors on trajectory datasets
   - Use curriculum learning for prediction horizon
   - Balance multi-modal diversity vs accuracy

2. **Evaluation Metrics**:
   - Monitor both ADE and FDE
   - Track mode diversity
   - Validate physical feasibility

3. **Deployment Considerations**:
   - Implement trajectory post-processing
   - Add collision checking
   - Consider computation budget