"""BEV feature analyzer for UniAD"""

from typing import Any, Dict, List

try:
    from ..core.data_structures import TraceNode
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode


class BEVFeatureTracer:
    """Specialized tracer for BEV transformations"""
    
    def __init__(self):
        self.bev_shape = (200, 200)  # Default UniAD BEV grid
        self.feature_dim = 256
    
    def analyze_bev_operations(self, trace_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Analyze BEV-specific operations"""
        bev_nodes = [node for node in trace_nodes if node.is_bev_operation]
        
        bev_stats = {
            'total_bev_ops': len(bev_nodes),
            'bev_memory_mb': sum(node.memory_usage for node in bev_nodes),
            'bev_compute_ms': sum(node.compute_time for node in bev_nodes),
            'grid_sizes': list(set(node.bev_grid_size for node in bev_nodes if node.bev_grid_size)),
            'frozen_encoder': any(node.is_frozen for node in bev_nodes)
        }
        
        return bev_stats