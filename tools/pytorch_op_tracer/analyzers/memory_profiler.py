"""Memory profiling for GPU memory usage"""

from typing import Any, Dict, List

try:
    from ..core.data_structures import TraceNode
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode


class MemoryProfiler:
    """Profile GPU memory usage"""
    
    def __init__(self, stage: int = 2):
        self.stage = stage
        self.expected_usage = {1: 30000, 2: 17000}  # MB
    
    def profile_memory(self, trace_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Profile memory usage and identify bottlenecks"""
        total_memory = sum(node.memory_usage for node in trace_nodes)
        
        # Find top memory consumers
        sorted_nodes = sorted(trace_nodes, key=lambda x: x.memory_usage, reverse=True)
        top_consumers = sorted_nodes[:10]
        
        memory_profile = {
            'total_memory_mb': total_memory,
            'expected_memory_mb': self.expected_usage.get(self.stage, 0),
            'memory_efficiency': (self.expected_usage.get(self.stage, total_memory) / total_memory) if total_memory > 0 else 0,
            'top_consumers': [
                {
                    'operation': node.operation,
                    'module': node.module_path,
                    'memory_mb': node.memory_usage,
                    'percentage': (node.memory_usage / total_memory * 100) if total_memory > 0 else 0
                }
                for node in top_consumers
            ]
        }
        
        return memory_profile