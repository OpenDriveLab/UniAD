# UniAD Attention Modules Report

## Overview

UniAD's attention mechanisms are the core components that enable effective multi-camera perception and temporal reasoning. The system uses two primary attention types: Temporal Self-Attention (TSA) and Spatial Cross-Attention (SCA).

## Temporal Self-Attention (TSA)

### Purpose
Aggregates temporal information from previous BEV features to maintain temporal consistency and leverage historical context.

### Architecture

```python
class TemporalSelfAttention(BaseModule):
    """
    Deformable attention for temporal BEV feature aggregation
    - embed_dims: 256
    - num_heads: 8
    - num_levels: 4 (multi-scale)
    - num_points: 4 (deformable sampling)
    - num_bev_queue: 2 (current + previous)
    """
```

### Key Mechanisms

#### 1. Temporal Feature Fusion
```
Current BEV Query + Previous BEV Features
            ↓
    Ego-motion Compensation
            ↓
    Deformable Attention
            ↓
    Temporal Aggregation
```

#### 2. Ego-Motion Compensation
- Aligns previous BEV features with current frame
- Handles vehicle movement between frames
- Uses CAN bus data for accurate transformation

#### 3. Deformable Sampling
- Learns optimal sampling locations in previous BEV
- 4 sampling points per query
- Multi-scale feature aggregation

### Implementation Details

```python
# Key forward pass logic
def forward(self, query, value, reference_points, ...):
    # Concatenate current and previous features
    query = torch.cat([value[:bs], query], -1)
    
    # Compute sampling offsets and attention weights
    sampling_offsets = self.sampling_offsets(query)
    attention_weights = self.attention_weights(query).softmax(-1)
    
    # Apply deformable attention
    output = MultiScaleDeformableAttnFunction.apply(
        value, spatial_shapes, level_start_index, 
        sampling_locations, attention_weights
    )
    
    # Fuse temporal features
    output = output.mean(-1)  # Average over temporal dimension
```

### Memory and Performance
- Input: BEV features (BS, 200×200, 256)
- Output: Temporally enhanced BEV (BS, 200×200, 256)
- Memory: ~2GB for queue length 2
- Computation: O(N×K) where N=40000 queries, K=4 points

## Spatial Cross-Attention (SCA)

### Purpose
Projects multi-camera 2D features into unified 3D BEV representation.

### Architecture

```python
class SpatialCrossAttention(BaseModule):
    """
    Camera-aware attention for BEV generation
    - embed_dims: 256
    - num_cams: 6
    - deformable_attention: MSDeformableAttention3D
    """
```

### Key Mechanisms

#### 1. Camera-Aware Processing
```
BEV Queries (3D points)
        ↓
Project to 2D camera coords
        ↓
Check visibility per camera
        ↓
Sample features from visible cameras
        ↓
Aggregate with attention weights
```

#### 2. Visibility Optimization
- Only processes BEV queries visible to each camera
- Significant memory savings (up to 60%)
- Dynamic batching by camera

#### 3. Multi-Height Sampling
- Each BEV query has 4 height anchors
- Samples features at different heights
- Handles occlusions and multi-level objects

### Implementation Details

```python
# Camera-aware batching
for j in range(bs):
    for i, reference_points_per_img in enumerate(reference_points_cam):
        # Only process visible queries
        index_query_per_img = indexes[i]
        queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]

# Deformable attention with 3D reference points
queries = self.deformable_attention(
    query=queries_rebatch,
    key=camera_features,
    value=camera_features,
    reference_points=reference_points_rebatch
)

# Normalize by visibility count
slots = slots / count[..., None]
```

## MSDeformableAttention3D

### Purpose
3D-aware deformable attention for efficient spatial sampling.

### Key Features

#### 1. 3D Reference Points
- Supports multiple Z-anchors per query
- Projects 3D points to 2D for each camera
- Handles perspective transformation

#### 2. Learned Sampling Offsets
```python
# For each BEV query with Z anchors
sampling_offsets = self.sampling_offsets(query)  # Learnable offsets
sampling_locations = reference_points + sampling_offsets

# Shape transformations
# (BS, num_query, num_heads, num_levels, num_points, num_Z_anchors, 2)
```

#### 3. Multi-Scale Processing
- 4 feature levels from FPN
- Different resolutions for coarse-to-fine processing
- Weighted aggregation across scales

### Performance Optimization

#### Memory Efficiency
1. **Visibility Masking**: 60% memory reduction
2. **Camera Batching**: Better GPU utilization
3. **Sparse Sampling**: Only 4-8 points per query

#### Computational Efficiency
1. **CUDA Kernels**: Custom deformable attention
2. **FP32 Operations**: Stability over FP16
3. **Batched Operations**: Minimize kernel launches

## Integration in BEVFormer

### Layer Structure
```
BEVFormerLayer:
1. Temporal Self-Attention (TSA)
   - Aggregate temporal information
2. Layer Norm
3. Spatial Cross-Attention (SCA)  
   - Project camera features to BEV
4. Layer Norm
5. Feed-Forward Network
6. Layer Norm
```

### Data Flow
```
Previous BEV ─┐
              ├─→ TSA ─→ Enhanced BEV ─→ SCA ─→ Final BEV
Current Query ─┘                          ↑
                                    Camera Features
```

## Configuration Parameters

### TSA Configuration
```yaml
temporal_self_attention:
  embed_dims: 256
  num_heads: 8
  num_levels: 4
  num_points: 4
  num_bev_queue: 2
  dropout: 0.1
```

### SCA Configuration
```yaml
spatial_cross_attention:
  embed_dims: 256
  num_cams: 6
  pc_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  deformable_attention:
    num_levels: 4
    num_points: 8
    num_z_anchors: 4
```

## Best Practices

### 1. Memory Management
- Use visibility masking for large scenes
- Reduce num_points for faster inference
- Enable gradient checkpointing if needed

### 2. Training Strategy
- Pre-train without temporal (TSA only)
- Gradually increase temporal context
- Use ego-motion augmentation

### 3. Deployment
- Cache camera projections
- Optimize CUDA kernels
- Consider TensorRT conversion

## Common Issues and Solutions

### 1. Memory Overflow
- Reduce batch size
- Decrease num_points
- Enable mixed precision (carefully)

### 2. Training Instability
- Use FP32 for attention computation
- Clip gradients
- Warm-up learning rate

### 3. Poor Temporal Consistency
- Check ego-motion compensation
- Verify CAN bus data quality
- Increase temporal supervision weight

The attention modules form the backbone of UniAD's perception system, enabling effective multi-camera 3D understanding with temporal consistency.