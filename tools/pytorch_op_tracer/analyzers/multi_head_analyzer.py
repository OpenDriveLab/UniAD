"""Multi-head interaction analyzer for UniAD"""

from collections import defaultdict
from typing import Any, Dict, List

try:
    from ..core.data_structures import TraceNode
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode


class MultiHeadTracer:
    """Traces interactions between UniAD's five task heads"""
    
    def __init__(self, task_heads: List[str] = None):
        self.task_heads = task_heads or ['track', 'seg', 'motion', 'occ', 'planning']
        self.head_dependencies = {
            'motion': ['track'],
            'occ': ['track'],
            'planning': ['track', 'motion', 'occ']
        }
    
    def analyze_head_interactions(self, trace_nodes: List[TraceNode]) -> Dict[str, Any]:
        """Analyze interactions between task heads"""
        head_nodes = defaultdict(list)
        head_memory = defaultdict(float)
        head_compute = defaultdict(float)
        
        # Group nodes by task head
        for node in trace_nodes:
            if node.task_head:
                head_nodes[node.task_head].append(node)
                head_memory[node.task_head] += node.memory_usage
                head_compute[node.task_head] += node.compute_time
        
        # Analyze dependencies
        dependency_flow = {}
        for head, deps in self.head_dependencies.items():
            if head in head_nodes:
                dependency_flow[head] = {
                    'depends_on': deps,
                    'node_count': len(head_nodes[head]),
                    'total_memory_mb': head_memory[head],
                    'total_compute_ms': head_compute[head]
                }
        
        return {
            'head_summary': dict(head_nodes),
            'memory_by_head': dict(head_memory),
            'compute_by_head': dict(head_compute),
            'dependency_flow': dependency_flow
        }