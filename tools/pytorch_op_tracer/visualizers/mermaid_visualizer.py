"""Mermaid diagram visualizer for dataflow"""

from collections import defaultdict
from typing import Any, Dict, List

try:
    from ..core.data_structures import TraceNode
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_structures import TraceNode


class DataflowVisualizer:
    """Generates Mermaid diagrams from trace data"""
    
    def __init__(self, max_nodes: int = 50):
        self.max_nodes = max_nodes
    
    def generate_mermaid(self, trace_nodes: List[TraceNode], 
                        head_analysis: Dict[str, Any],
                        memory_profile: Dict[str, Any]) -> str:
        """Generate Mermaid diagram from trace data"""
        lines = ["```mermaid", "graph TB"]
        
        # Add temporal queue subgraph
        lines.append('    subgraph "Temporal Queue"')
        lines.append('        Input[Multi-Frame Input] --> BEV[BEV Features]')
        lines.append('    end')
        lines.append('')
        
        # Add task heads subgraph
        lines.append('    subgraph "Task Heads"')
        
        # Group nodes by task head
        head_nodes = defaultdict(list)
        for node in trace_nodes[:self.max_nodes]:
            if node.task_head:
                head_nodes[node.task_head].append(node)
        
        # Create simplified nodes for each head
        head_info = head_analysis.get('memory_by_head', {})
        compute_info = head_analysis.get('compute_by_head', {})
        
        for head in ['track', 'seg', 'motion', 'occ', 'planning']:
            if head in head_nodes:
                memory = head_info.get(head, 0)
                compute = compute_info.get(head, 0)
                node_id = head.capitalize()
                label = f"{node_id} Head<br/>Memory: {memory:.1f}MB<br/>Time: {compute:.1f}ms"
                lines.append(f'        {node_id}["{label}"]')
        
        lines.append('    end')
        lines.append('')
        
        # Add dependencies
        lines.append('    BEV --> Track')
        lines.append('    BEV --> Seg')
        lines.append('    Track --> Motion')
        lines.append('    Track --> Occ')
        lines.append('    Motion --> Planning')
        lines.append('    Occ --> Planning')
        lines.append('')
        
        # Add styling
        lines.append('    style Track fill:#f9f,stroke:#333,stroke-width:2px')
        lines.append('    style Motion fill:#bbf,stroke:#333,stroke-width:2px')
        lines.append('    style Planning fill:#bfb,stroke:#333,stroke-width:2px')
        
        lines.append("```")
        
        return '\n'.join(lines)
    
    def generate_memory_heatmap(self, memory_profile: Dict[str, Any]) -> str:
        """Generate memory usage heatmap"""
        lines = ["### Memory Usage Heatmap", ""]
        lines.append("```")
        lines.append(f"{'Operation':<30} | {'Memory (MB)':<12} | {'Percentage':<10} | {'Visual':<40}")
        lines.append("-" * 95)
        
        total_memory = memory_profile['total_memory_mb']
        for consumer in memory_profile['top_consumers']:
            op = consumer['operation'][:30]
            mem = consumer['memory_mb']
            pct = consumer['percentage']
            bar_length = int(pct / 100 * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            
            lines.append(f"{op:<30} | {mem:<12.1f} | {pct:<10.1f} | {bar}")
        
        lines.append("-" * 95)
        lines.append(f"{'Total':<30} | {total_memory:<12.1f} | {'100.0':<10} | {'█' * 40}")
        lines.append("```")
        
        return '\n'.join(lines)