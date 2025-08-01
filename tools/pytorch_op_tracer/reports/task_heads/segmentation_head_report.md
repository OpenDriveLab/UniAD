# UniAD Segmentation Head Module Analysis Report

## Overview

The Segmentation Head (PansegformerHead) provides bird's-eye view (BEV) semantic segmentation for road understanding, including lanes, walkways, and drivable areas. It uses a panoptic segmentation approach to handle both stuff (background) and thing (instance) classes.

## Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                     PansegformerHead                          │
├──────────────────────────────────────────────────────────────┤
│ • BEV Grid: 200×200 @ 0.512m/pixel                          │
│ • Classes: lanes, walkways, vehicles, etc.                   │
│ • Panoptic approach: stuff + thing segmentation              │
│ • Multi-scale decoder with FPN                               │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  Segmentation Pipeline                        │
├──────────────────────────────────────────────────────────────┤
│  BEV Features → Transformer Decoder → Mask Prediction        │
│       ↓              ↓                    ↓                  │
│  Positional     Query-based         Pixel-level              │
│  Encoding       Attention           Classification            │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Panoptic Segmentation Design

The head handles both:
- **Stuff classes**: Road surfaces, lanes, walkways (semantic segmentation)
- **Thing classes**: Vehicles, pedestrians (instance segmentation)

### 2. Query-Based Architecture

```python
# Uses learnable queries for different semantic regions
num_queries = 100  # Learnable semantic queries
query_dim = 256    # Query embedding dimension

# Transformer decoder for query refinement
decoder_layers = 6
decoder = TransformerDecoder(
    num_layers=decoder_layers,
    hidden_dim=query_dim,
    num_heads=8
)
```

### 3. Multi-Scale Processing

```python
# Feature Pyramid Network (FPN) for multi-scale features
fpn_levels = [1/8, 1/16, 1/32]  # Multiple resolutions
fpn = FPN(
    in_channels=[256, 512, 1024],
    out_channels=256
)
```

## Segmentation Pipeline

### Input Processing

| Input | Shape | Description |
|-------|-------|-------------|
| bev_embed | (B, 256, 200, 200) | BEV features from encoder |
| bev_h, bev_w | 200, 200 | BEV grid dimensions |
| bev_pos | (1, 256, 200, 200) | Positional embeddings |

### Processing Flow

```python
# 1. Initialize semantic queries
query_embeds = self.query_embed.weight  # (100, 256)
query_pos = self.query_pos.weight      # (100, 256)

# 2. Decode through transformer
hs = self.transformer_decoder(
    query=query_embeds,
    key=bev_embed.flatten(2),
    value=bev_embed.flatten(2),
    query_pos=query_pos,
    key_pos=bev_pos.flatten(2)
)

# 3. Generate class predictions and masks
outputs_class = self.class_embed(hs)     # (B, 100, num_classes)
outputs_mask = self.mask_embed(hs)       # (B, 100, H*W)

# 4. Reshape masks to spatial dimensions
outputs_mask = outputs_mask.view(B, 100, H, W)
```

## Loss Functions

### 1. Classification Loss

```python
# Focal loss for class predictions
class_loss = sigmoid_focal_loss(
    predictions=outputs_class,
    targets=gt_labels,
    alpha=0.25,
    gamma=2.0
)
```

### 2. Mask Loss

```python
# Combination of Dice loss and Focal loss for masks
mask_loss = dice_loss(pred_masks, gt_masks) + \
            sigmoid_focal_loss(pred_masks, gt_masks)
```

### 3. Hungarian Matching

```python
# Optimal assignment between predictions and ground truth
matcher = HungarianMatcher(
    cost_class=1.0,
    cost_mask=5.0,
    cost_dice=2.0
)
indices = matcher(outputs_class, outputs_mask, gt_labels, gt_masks)
```

## Integration with UniAD

### Dependencies

- **BEV Encoder**: Provides spatial features
- **Track Head**: Shares BEV features (no direct dependency)

### Output Interface

```python
seg_results = {
    "seg_preds": pred_masks,        # (B, H, W, C) segmentation maps
    "seg_scores": class_scores,     # (B, N, C) class confidences
    "bev_embed": bev_embed,         # Pass through for other heads
    "bev_pos": bev_pos              # Positional encoding
}
```

## Performance Characteristics

### Metrics
- **mIoU**: 63.7% on nuScenes validation
- **Classes**: ~10-15 semantic classes
- **Resolution**: 200×200 BEV grid

### Computational Requirements
- **Memory**: ~5 GB
- **Inference Time**: ~20ms
- **FLOPs**: ~10 GFLOPs

## Configuration

```yaml
# Segmentation head configuration
num_classes: 10
num_queries: 100
num_decoder_layers: 6
hidden_dim: 256
num_heads: 8

# Loss weights
loss_weight_class: 1.0
loss_weight_mask: 5.0
loss_weight_dice: 2.0

# BEV grid
bev_h: 200
bev_w: 200
pc_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
```

## Key Algorithms

### 1. Panoptic Post-Processing

```python
# Merge stuff and thing predictions
def panoptic_fusion(semantic_seg, instance_seg):
    # Stuff classes: direct semantic segmentation
    panoptic = semantic_seg.copy()
    
    # Thing classes: instance segmentation
    for instance in instances:
        mask = instance['mask'] > threshold
        panoptic[mask] = instance['class'] + instance['id'] * 1000
    
    return panoptic
```

### 2. Multi-Scale Feature Aggregation

```python
# FPN for combining multi-resolution features
def multi_scale_features(feat_list):
    # Upsample and combine
    combined = 0
    for feat, scale in zip(feat_list, scales):
        feat_up = F.interpolate(feat, size=target_size)
        combined += feat_up * scale
    return combined
```

## Best Practices

1. **Training Strategy**:
   - Use auxiliary losses at multiple decoder layers
   - Apply deep supervision for better gradients
   - Balance stuff vs thing class weights

2. **Data Augmentation**:
   - Random BEV rotation
   - Color jittering for robustness
   - Copy-paste augmentation for things

3. **Inference Optimization**:
   - Cache query embeddings
   - Use TensorRT for deployment
   - Implement sliding window for large scenes