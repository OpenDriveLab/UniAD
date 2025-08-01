"""
PyTorch Operation Tracer for UniAD

A comprehensive tool for tracing PyTorch operations, recording tensor shapes,
and visualizing dataflow with specific support for UniAD's multi-task architecture.
"""

__version__ = "1.0.0"
__author__ = "UniAD PyTorch Tracer Team"

from .core.tracer import OperationTracer
from .core.data_structures import TraceNode
from .analyzers.trace_analyzer import TraceAnalyzer
from .visualizers.mermaid_visualizer import DataflowVisualizer

__all__ = [
    "OperationTracer",
    "TraceNode",
    "TraceAnalyzer",
    "DataflowVisualizer",
]