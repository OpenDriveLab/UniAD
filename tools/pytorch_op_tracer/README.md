# PyTorch Operation Tracer

A comprehensive tool for tracing PyTorch operations, recording tensor shapes, and visualizing dataflow with specific support for UniAD's multi-task autonomous driving framework.

## Features

- **Operation Tracing**: Hooks into PyTorch modules to trace all operations
- **UniAD-Specific Support**: Specialized tracking for UniAD's 5 task heads (track, seg, motion, occ, planning)
- **Memory Profiling**: Critical for understanding UniAD's 30-50GB GPU memory usage
- **Temporal Analysis**: Visualize multi-frame temporal queue processing
- **BEV Feature Tracking**: Specialized analysis for BEV encoder/decoder operations
- **Mermaid Visualization**: Generate clear dataflow diagrams
- **Stage-Aware**: Different handling for Stage 1 (perception) vs Stage 2 (end-to-end)

## Installation

### Option 1: Install as Package

```bash
# Clone the repository
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD/tools/pytorch_op_tracer

# Install the package
pip install -e .

# With visualization support
pip install -e ".[visualization]"

# With full UniAD support
pip install -e ".[uniad]"
```

### Option 2: Use Directly

```bash
# Add to Python path
export PYTHONPATH=$PYTHONPATH:/path/to/UniAD/tools/pytorch_op_tracer

# Run directly
python /path/to/UniAD/tools/pytorch_op_tracer/trace_ops.py --help
```

## Quick Start

### Basic Usage

```bash
# Trace a UniAD model
pytorch-trace --config projects/configs/stage2_e2e/base_e2e.py \
              --checkpoint ckpts/uniad_base_e2e.pth \
              --output trace_output.md

# Test mode without UniAD dependencies
pytorch-trace --test-mode --output test_trace.md
```

### Python API

```python
from pytorch_op_tracer import OperationTracer, TraceAnalyzer, DataflowVisualizer

# Create tracer
tracer = OperationTracer(model, stage=2, task_heads=['track', 'motion', 'planning'])

# Trace model
trace_nodes = tracer.trace(inputs)

# Analyze
analyzer = TraceAnalyzer()
analysis = analyzer.analyze(trace_nodes)

# Visualize
visualizer = DataflowVisualizer()
mermaid = visualizer.generate_mermaid(trace_nodes, analysis['head_analysis'], 
                                     analysis['memory_profile'])
```

## Package Structure

```
pytorch_op_tracer/
├── __init__.py              # Package root
├── core/                    # Core tracing functionality
│   ├── __init__.py
│   ├── data_structures.py   # TraceNode data structure
│   ├── shape_recorder.py    # Tensor shape recording
│   └── tracer.py           # Main OperationTracer
├── analyzers/              # Analysis modules
│   ├── __init__.py
│   ├── trace_analyzer.py   # Comprehensive analyzer
│   ├── multi_head_analyzer.py  # UniAD task head analysis
│   ├── temporal_analyzer.py    # Temporal queue analysis
│   ├── bev_analyzer.py        # BEV feature analysis
│   └── memory_profiler.py     # Memory profiling
├── visualizers/            # Visualization modules
│   ├── __init__.py
│   └── mermaid_visualizer.py  # Mermaid diagram generation
├── utils/                  # Utilities
│   ├── __init__.py
│   └── model_utils.py      # Model loading utilities
├── examples/               # Example scripts
├── tests/                  # Test suite
├── trace_ops.py           # Main CLI script
├── setup.py               # Package setup
└── README.md              # This file
```

## Command Line Options

### Model Configuration
- `--config`: Path to UniAD config file (required)
- `--checkpoint`: Path to model checkpoint
- `--stage`: UniAD stage (1: perception, 2: end-to-end)
- `--test-mode`: Run with dummy model for testing

### Tracing Options
- `--task-heads`: Task heads to trace (default: all)
- `--temporal-frames`: Number of temporal frames (default: 3)
- `--trace-backward`: Trace backward pass
- `--filter-ops`: Filter specific operations
- `--max-nodes`: Maximum nodes to visualize (default: 50)

### Analysis Options
- `--memory-profile`: Enable detailed memory profiling
- `--bev-focus`: Focus analysis on BEV operations
- `--visualize-temporal`: Visualize temporal flow

### Output Options
- `--output`: Output file path (default: trace_output.md)
- `--export-json`: Export trace data as JSON
- `--device`: Device to run on (cuda/cpu)

## Usage Examples

### Example 1: Trace UniAD Stage 2
```bash
pytorch-trace --config projects/configs/stage2_e2e/base_e2e.py \
              --checkpoint ckpts/uniad_base_e2e.pth \
              --stage 2 \
              --task-heads track motion planning \
              --memory-profile \
              --output stage2_analysis.md
```

### Example 2: Focus on BEV Operations
```bash
pytorch-trace --config projects/configs/stage2_e2e/base_e2e.py \
              --checkpoint ckpts/uniad_base_e2e.pth \
              --bev-focus \
              --filter-ops BEVFormer BEVEncoder \
              --output bev_analysis.md
```

### Example 3: Export for Custom Analysis
```bash
pytorch-trace --config projects/configs/stage2_e2e/base_e2e.py \
              --checkpoint ckpts/uniad_base_e2e.pth \
              --export-json \
              --output trace_data.md

# Analyze in Python
import json
import pandas as pd

with open('trace_data.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['nodes'])
print(df.groupby('task_head')['memory_usage'].sum())
```

## Output Format

The tool generates a comprehensive Markdown report containing:

1. **Summary Statistics**: Total operations, memory usage, compute time
2. **Task Head Analysis**: Memory and compute breakdown by task head
3. **Dataflow Visualization**: Mermaid diagram showing task dependencies
4. **Memory Heatmap**: Visual representation of memory consumers
5. **JSON Export**: Detailed trace data for custom analysis

## Development

### Running Tests
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run specific test
python tests/test_tracer.py
```

### Code Style
```bash
# Format code
black pytorch_op_tracer/

# Check style
flake8 pytorch_op_tracer/
```

## Troubleshooting

### ImportError: mmdet3d not available
- Install UniAD dependencies: `pip install -e ".[uniad]"`
- Or use `--test-mode` for testing without UniAD

### Out of Memory
- Use `--filter-ops` to trace specific operations
- Reduce `--max-nodes` for visualization
- Run on CPU with `--device cpu`

### Module Not Found
- Ensure package is installed: `pip install -e .`
- Or add to PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:/path/to/pytorch_op_tracer`

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This tool is part of the UniAD project and follows the same Apache 2.0 license.