"""Data structures for PyTorch Operation Tracer"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TraceNode:
    """Enhanced data structure for tracing operations in UniAD"""
    
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
    
    # Unique identifier
    node_id: str = ""
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"{self.module_path}_{id(self)}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'operation': self.operation,
            'module_path': self.module_path,
            'input_shapes': self.input_shapes,
            'output_shapes': self.output_shapes,
            'task_head': self.task_head,
            'temporal_index': self.temporal_index,
            'is_frozen': self.is_frozen,
            'memory_usage': self.memory_usage,
            'compute_time': self.compute_time,
            'flops': self.flops,
            'is_bev_operation': self.is_bev_operation,
            'bev_grid_size': self.bev_grid_size,
            'depends_on': self.depends_on,
            'feeds_into': self.feeds_into,
            'node_id': self.node_id
        }