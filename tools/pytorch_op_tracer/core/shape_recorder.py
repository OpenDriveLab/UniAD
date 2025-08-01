"""Tensor shape recording utilities"""

from typing import Any, List, Tuple
import torch


class TensorShapeRecorder:
    """Records shapes of tensors at each operation"""
    
    @staticmethod
    def extract_shapes(tensors: Any) -> List[Tuple]:
        """Extract shapes from various tensor containers"""
        if isinstance(tensors, torch.Tensor):
            return [tuple(tensors.shape)]
        elif isinstance(tensors, (list, tuple)):
            shapes = []
            for t in tensors:
                shapes.extend(TensorShapeRecorder.extract_shapes(t))
            return shapes
        elif isinstance(tensors, dict):
            shapes = []
            for v in tensors.values():
                shapes.extend(TensorShapeRecorder.extract_shapes(v))
            return shapes
        else:
            return []
    
    @staticmethod
    def estimate_memory(shapes: List[Tuple], dtype=torch.float32) -> float:
        """Estimate memory usage in MB"""
        bytes_per_element = 4 if dtype == torch.float32 else 2  # assuming fp32 or fp16
        total_elements = sum(torch.prod(torch.tensor(shape)).item() for shape in shapes)
        return (total_elements * bytes_per_element) / (1024 * 1024)