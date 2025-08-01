# PyTorch Operation Tracer and Visualizer for UniAD

This document outlines the architecture design for a script that traces PyTorch operations, records tensor shapes, and visualizes the dataflow using Mermaid diagrams, with specific enhancements for the UniAD multi-task autonomous driving framework.

## 1. Overview

The PyTorch Operation Tracer is designed to help developers understand the flow of tensors through neural network models, with specialized support for UniAD's hierarchical multi-task architecture:

1. Tracing all PyTorch operations during model execution
2. Recording input and output tensor shapes for each operation
3. Visualizing the dataflow as a Mermaid diagram with task-specific views
4. Providing detailed analysis of the model's computational graph
5. **UniAD-specific**: Tracking multi-head interactions, temporal queues, and BEV feature flow
6. **Memory profiling**: Essential for UniAD's 30-50GB GPU memory requirements

## 2. Architecture Components

### 2.1 Core Components

The system consists of the following core components:

1. **OperationTracer**: Hooks into PyTorch modules to trace operations
2. **TensorShapeRecorder**: Records shapes of input/output tensors
3. **DataflowVisualizer**: Generates Mermaid diagrams from trace data
4. **TraceAnalyzer**: Analyzes the trace data for insights
5. **CommandLineInterface**: Provides a user-friendly interface

#### UniAD-Specific Components:

6. **MultiHeadTracer**: Traces UniAD's five task heads (track, seg, motion, occ, planning) and their interactions
7. **TemporalTracer**: Handles temporal queue operations (3-5 frames)
8. **BEVFeatureTracer**: Specialized for BEV encoder/decoder transformations
9. **MemoryProfiler**: Tracks GPU memory usage patterns (critical for 30-50GB requirements)
10. **TaskDependencyAnalyzer**: Analyzes dependencies between task heads

### 2.2 Component Interactions

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│ CommandLineUI   │────▶│ OperationTracer   │────▶│ TensorShapeRecorder │
└─────────────────┘     └───────────────────┘     └─────────────────────┘
                                 │                            │
                                 ├──────────┐                 │
                                 ▼          ▼                 ▼
                         ┌─────────────┐ ┌──────────────┐ ┌───────────────┐
                         │MultiHeadTrace│ │TemporalTrace │ │ Trace Data    │
                         └─────────────┘ └──────────────┘ └───────────────┘
                                 │          │                 │
                                 ▼          ▼                 ▼
                         ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
                         │ TraceAnalyzer │ │MemoryProfile│ │BEVFeatureTrace│
                         └───────────────┘ └──────────────┘ └──────────────┘
                                 │                            ▲
                                 ▼                            │
                        ┌────────────────┐                    │
                        │ DataflowVisual │────────────────────┘
                        └────────────────┘
```

## 3. Detailed Component Design

### 3.1 OperationTracer

The OperationTracer uses PyTorch hooks to intercept forward and backward passes through the model:

- **Module Hooks**: Register hooks on PyTorch modules
- **Function Hooks**: Register hooks on PyTorch autograd functions
- **Tensor Hooks**: Track tensor operations

Key features:
- Track execution order of operations
- Identify connections between operations
- Support for custom filtering of operations

```python
class OperationTracer:
    def __init__(self, model, trace_backward=False, filter_ops=None,
                 stage=2, task_heads=None):
        # Initialize tracer with UniAD-specific options
        self.stage = stage  # Stage 1 or 2
        self.task_heads = task_heads or ['track', 'seg', 'motion', 'occ', 'planning']

    def register_hooks(self):
        # Register hooks on model modules

    def trace(self, inputs):
        # Perform tracing with given inputs

    def get_trace_data(self):
        # Return collected trace data with task head annotations
```

### 3.2 TensorShapeRecorder

Records the shapes of tensors at each operation:

- Input tensor shapes
- Output tensor shapes
- Parameter shapes

Handles various tensor containers (lists, tuples, dictionaries).

#### Enhanced Data Structure for UniAD:

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class TraceNode:
    # Basic information
    operation: str
    module_path: str
    input_shapes: List[Tuple]
    output_shapes: List[Tuple]

    # UniAD-specific fields
    task_head: Optional[str] = None  # track/seg/motion/occ/planning
    temporal_index: Optional[int] = None  # frame index in queue
    is_frozen: bool = False  # for frozen BEV encoder in stage 2

    # Performance metrics
    memory_usage: float = 0.0  # MB
    compute_time: float = 0.0  # ms
    flops: Optional[int] = None

    # BEV-specific
    is_bev_operation: bool = False
    bev_grid_size: Optional[Tuple[int, int]] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    feeds_into: List[str] = field(default_factory=list)
```

### 3.3 DataflowVisualizer

Generates visualizations from the trace data:

- **Mermaid Diagram Generator**: Creates Mermaid syntax for dataflow
- **Graph Simplification**: Simplifies complex graphs for readability
- **Hierarchical Grouping**: Groups operations by module hierarchy

Features:
- Customizable node appearance
- Filtering options for large models
- Support for exporting to various formats

### 3.4 TraceAnalyzer

Analyzes the trace data to provide insights:

- Memory usage estimation
- Computational complexity analysis
- Bottleneck identification
- Layer-wise timing analysis

### 3.5 CommandLineInterface

Provides a user-friendly interface:

```
python tools/analysis_tools/trace_pytorch_ops.py \
    --config CONFIG_PATH \
    --checkpoint CHECKPOINT_PATH \
    --input-shape 1,3,224,224 \
    --output mermaid_diagram.md
```

Options:
- Model configuration
- Input specifications
- Visualization options
- Analysis preferences

### 3.6 UniAD-Specific Components

#### 3.6.1 MultiHeadTracer

Traces interactions between UniAD's five task heads:

```python
class MultiHeadTracer:
    def __init__(self, task_heads=['track', 'seg', 'motion', 'occ', 'planning']):
        self.task_heads = task_heads
        self.head_dependencies = {
            'motion': ['track'],
            'occ': ['track'],
            'planning': ['track', 'motion', 'occ']
        }

    def trace_task_heads(self, model):
        # Track data flow between task heads
        # Identify inter-head dependencies
        pass
```

#### 3.6.2 TemporalTracer

Handles temporal queue operations for multi-frame processing:

```python
class TemporalTracer:
    def __init__(self, queue_length=3):
        self.queue_length = queue_length

    def trace_temporal_flow(self, bev_features):
        # Track how features aggregate over frames
        # Visualize temporal fusion operations
        pass
```

#### 3.6.3 BEVFeatureTracer

Specialized tracer for BEV transformations:

```python
class BEVFeatureTracer:
    def __init__(self):
        self.bev_shape = (200, 200)  # BEV grid size
        self.feature_dim = 256

    def trace_bev_operations(self, encoder, decoder):
        # Track BEV encoder operations
        # Monitor feature propagation through decoder
        # Identify frozen vs trainable components
        pass
```

#### 3.6.4 MemoryProfiler

Profile GPU memory usage:

```python
class MemoryProfiler:
    def __init__(self, stage=2):
        self.stage = stage
        self.expected_usage = {1: 30000, 2: 17000}  # MB

    def profile_gpu_memory(self):
        # Track memory allocation per operation
        # Identify memory bottlenecks
        # Compare against expected usage
        pass
```

## 4. Implementation Plan

### 4.1 Phase 1: Core Tracing with UniAD Awareness

1. Implement basic module hooks with task head detection
2. Record tensor shapes with BEV grid awareness
3. Track execution order and task dependencies
4. Add stage-specific tracing (Stage 1 vs Stage 2)

### 4.2 Phase 2: Temporal and Multi-Head Support

1. Implement temporal queue tracing
2. Add multi-head interaction tracking
3. Visualize task hierarchy (perception → prediction → planning)
4. Track frozen vs trainable components

### 4.3 Phase 3: Memory Profiling and Optimization

1. Implement GPU memory profiling
2. Identify memory bottlenecks (critical for 30-50GB usage)
3. Add memory-efficient tracing modes
4. Compare memory usage between stages

### 4.4 Phase 4: Advanced Visualization

1. Generate task-aware Mermaid diagrams
2. Add temporal dimension visualization
3. Create memory heatmaps
4. Implement hierarchical task views

### 4.5 Phase 5: Analysis and Optimization

1. Task dependency analysis
2. Performance bottleneck identification
3. Stage comparison tools
4. Optimization recommendations

## 5. Usage Examples

### 5.1 Basic Usage

```bash
# Trace UniAD Stage 1 (perception only)
python tools/analysis_tools/trace_pytorch_ops.py \
    --config projects/configs/stage1_track_map/base_track_map.py \
    --checkpoint ckpts/uniad_base_track_map.pth \
    --stage 1

# Trace UniAD Stage 2 (end-to-end)
python tools/analysis_tools/trace_pytorch_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --stage 2
```

### 5.2 Advanced Usage

```bash
# Trace specific task heads with memory profiling
python tools/analysis_tools/trace_pytorch_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --task-heads track,motion,planning \
    --temporal-frames 3 \
    --memory-profile \
    --output uniad_trace_analysis.md

# Focus on BEV operations
python tools/analysis_tools/trace_pytorch_ops.py \
    --config projects/configs/stage2_e2e/base_e2e.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --bev-focus \
    --filter-ops BEVFormer,BEVEncoder,BEVDecoder \
    --visualize-temporal \
    --output bev_flow.md

# Compare Stage 1 and Stage 2
python tools/analysis_tools/trace_pytorch_ops.py \
    --compare-stages \
    --stage1-config projects/configs/stage1_track_map/base_track_map.py \
    --stage1-ckpt ckpts/uniad_base_track_map.pth \
    --stage2-config projects/configs/stage2_e2e/base_e2e.py \
    --stage2-ckpt ckpts/uniad_base_e2e.pth \
    --output stage_comparison.md
```

## 6. Challenges and Considerations

### 6.1 Performance Impact

- Tracing adds overhead to model execution
- Memory usage can be significant for large models

### 6.2 Complex Models

- Handling recurrent connections
- Dealing with dynamic control flow
- Supporting custom PyTorch extensions

### 6.3 Visualization Complexity

- Large models produce complex diagrams
- Need effective simplification strategies
- Balance between detail and readability

### 6.4 UniAD-Specific Challenges

- **Memory Management**: Handling 30-50GB GPU memory requirements
- **Temporal Complexity**: Visualizing multi-frame aggregation
- **Task Dependencies**: Tracking complex inter-head dependencies
- **Stage Differences**: Different architectures between Stage 1 and 2

## 7. Sample Visualization Output

### 7.1 Task Head Flow Diagram

```mermaid
graph TB
    subgraph "Temporal Queue (t-2 to t)"
        I1[Image t-2<br/>3x928x1600] --> BEV1[BEV t-2<br/>256x200x200]
        I2[Image t-1<br/>3x928x1600] --> BEV2[BEV t-1<br/>256x200x200]
        I3[Image t<br/>3x928x1600] --> BEV3[BEV t<br/>256x200x200]
    end

    subgraph "Task Heads"
        BEV3 --> Track[Track Head<br/>AMOTA: 0.390<br/>Memory: 8GB]
        BEV3 --> Seg[Seg Head<br/>IoU: 63.7%<br/>Memory: 5GB]
        Track --> Motion[Motion Head<br/>minADE: 0.705<br/>Memory: 4GB]
        Track --> Occ[Occ Head<br/>IoU: 63.7%<br/>Memory: 6GB]
        Motion --> Plan[Planning Head<br/>Col Rate: 0.29%<br/>Memory: 3GB]
        Occ --> Plan
    end

    style Track fill:#f9f,stroke:#333,stroke-width:2px
    style Motion fill:#bbf,stroke:#333,stroke-width:2px
    style Plan fill:#bfb,stroke:#333,stroke-width:2px
```

### 7.2 Memory Usage Heatmap

```
Operation               | Memory (GB) | Percentage | Visual
------------------------|-------------|------------|--------
BEV Encoder            | 15.2        | 30.4%      | ████████████████████████████████░░░░░░░░
Track Head             | 8.0         | 16.0%      | ████████████████░░░░░░░░░░░░░░░░░░░░░░░░
Temporal Aggregation   | 7.5         | 15.0%      | ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░
Seg Head               | 5.0         | 10.0%      | ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Occ Head               | 6.0         | 12.0%      | ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Motion Head            | 4.0         | 8.0%       | ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Planning Head          | 3.0         | 6.0%       | ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Others                 | 1.3         | 2.6%       | ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Total                  | 50.0        | 100%       | ████████████████████████████████████████
```

## 8. Future Extensions

- Integration with profiling tools (PyTorch Profiler, NVIDIA Nsight)
- Support for distributed training analysis
- Interactive web-based visualization
- Real-time tracing during training
- Automatic optimization suggestions based on bottlenecks
- Integration with TensorBoard for comprehensive analysis
- Support for custom UniAD variants and configurations
