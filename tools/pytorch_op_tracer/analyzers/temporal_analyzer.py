"""Temporal queue analyzer for multi-frame processing"""

from collections import defaultdict
from typing import Any, Dict, List

try:
    from ..core.data_structures import TraceNode
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode


class TemporalTracer:
    """Handles temporal queue operations for multi-frame processing"""
    
    def __init__(self, queue_length: int = 3):
        self.queue_length = queue_length
    
    def analyze_temporal_flow(self, trace_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Analyze temporal aggregation in the model"""
        temporal_nodes = [node for node in trace_nodes if node.temporal_index is not None]
        
        frames_data = defaultdict(lambda: {'nodes': [], 'memory': 0, 'compute': 0})
        for node in temporal_nodes:
            frame_idx = node.temporal_index
            frames_data[frame_idx]['nodes'].append(node)
            frames_data[frame_idx]['memory'] += node.memory_usage
            frames_data[frame_idx]['compute'] += node.compute_time
        
        return {
            'queue_length': self.queue_length,
            'frames_analyzed': len(frames_data),
            'temporal_memory_mb': sum(f['memory'] for f in frames_data.values()),
            'temporal_compute_ms': sum(f['compute'] for f in frames_data.values()),
            'frame_details': dict(frames_data)
        }