"""Comprehensive trace analyzer"""

from typing import Any, Dict, List

try:
    from ..core.data_structures import TraceNode
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode
from .multi_head_analyzer import MultiHeadTracer
from .temporal_analyzer import TemporalTracer
from .bev_analyzer import BEVFeatureTracer
from .memory_profiler import MemoryProfiler


class TraceAnalyzer:
    """Comprehensive analysis of trace data"""
    
    def __init__(self):
        self.multi_head_tracer = MultiHeadTracer()
        self.temporal_tracer = TemporalTracer()
        self.bev_tracer = BEVFeatureTracer()
        self.memory_profiler = MemoryProfiler()
    
    def analyze(self, trace_nodes: List[TraceNode], stage: int = 2) -> Dict[str, Any]:
        """Perform comprehensive analysis"""
        self.memory_profiler.stage = stage
        
        analysis = {
            'summary': {
                'total_operations': len(trace_nodes),
                'total_memory_mb': sum(node.memory_usage for node in trace_nodes),
                'total_compute_ms': sum(node.compute_time for node in trace_nodes),
                'unique_operations': len(set(node.operation for node in trace_nodes))
            },
            'head_analysis': self.multi_head_tracer.analyze_head_interactions(trace_nodes),
            'temporal_analysis': self.temporal_tracer.analyze_temporal_flow(trace_nodes),
            'bev_analysis': self.bev_tracer.analyze_bev_operations(trace_nodes),
            'memory_profile': self.memory_profiler.profile_memory(trace_nodes)
        }
        
        return analysis