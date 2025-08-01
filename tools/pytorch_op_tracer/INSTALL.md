# Installation Guide for PyTorch Operation Tracer

## Quick Start

### Option 1: Use without installation

```bash
# Navigate to the package directory
cd /path/to/UniAD/tools/pytorch_op_tracer

# Run directly
python trace_ops.py --test-mode --device cpu --output test.md
```

### Option 2: Install as a package

```bash
# Navigate to the package directory
cd /path/to/UniAD/tools/pytorch_op_tracer

# Install in development mode
pip install -e .

# Now you can use the command from anywhere
pytorch-trace --test-mode --device cpu --output test.md
```

### Option 3: Add to Python path

```bash
# Add to your ~/.bashrc or ~/.zshrc
export PYTHONPATH=$PYTHONPATH:/path/to/UniAD/tools/pytorch_op_tracer

# Then run from anywhere
python -m pytorch_op_tracer.trace_ops --test-mode --output test.md
```

## Full Installation with UniAD Support

```bash
# Install with all dependencies
pip install -e ".[uniad,visualization]"
```

## Verify Installation

```bash
# Test basic functionality
python trace_ops.py --test-mode --device cpu --output test.md

# Run tests
python tests/test_tracer.py

# Check help
python trace_ops.py --help
```

## Troubleshooting

### CUDA not available
Use `--device cpu` flag when running on systems without CUDA.

### Module not found errors
Make sure you're in the pytorch_op_tracer directory or have installed the package.

### UniAD dependencies missing
The tracer works in limited mode without mmdet3d. Use `--test-mode` for testing without UniAD.