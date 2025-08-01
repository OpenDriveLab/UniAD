#!/usr/bin/env python3
"""
Test script for the PyTorch Operation Tracer
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core import OperationTracer, TensorShapeRecorder
from core.data_structures import TraceNode
from analyzers import TraceAnalyzer
from visualizers import DataflowVisualizer


def test_tensor_shape_recorder():
    """Test TensorShapeRecorder functionality"""
    print("Testing TensorShapeRecorder...")
    
    # Test single tensor
    tensor = torch.randn(2, 3, 224, 224)
    shapes = TensorShapeRecorder.extract_shapes(tensor)
    assert shapes == [(2, 3, 224, 224)], f"Expected [(2, 3, 224, 224)], got {shapes}"
    
    # Test list of tensors
    tensor_list = [torch.randn(2, 3), torch.randn(4, 5)]
    shapes = TensorShapeRecorder.extract_shapes(tensor_list)
    assert shapes == [(2, 3), (4, 5)], f"Expected [(2, 3), (4, 5)], got {shapes}"
    
    # Test memory estimation
    memory = TensorShapeRecorder.estimate_memory([(2, 3, 224, 224)])
    expected = 2 * 3 * 224 * 224 * 4 / (1024 * 1024)  # MB
    assert abs(memory - expected) < 0.01, f"Expected ~{expected:.2f} MB, got {memory:.2f} MB"
    
    print("✓ TensorShapeRecorder tests passed")


def test_trace_node():
    """Test TraceNode dataclass"""
    print("\nTesting TraceNode...")
    
    node = TraceNode(
        operation="Conv2d",
        module_path="model.conv1",
        input_shapes=[(1, 3, 224, 224)],
        output_shapes=[(1, 64, 224, 224)],
        task_head="track",
        memory_usage=12.5
    )
    
    assert node.operation == "Conv2d"
    assert node.task_head == "track"
    assert node.memory_usage == 12.5
    assert node.node_id != ""  # Should be auto-generated
    
    print("✓ TraceNode tests passed")


def test_operation_tracer():
    """Test OperationTracer with a simple model"""
    print("\nTesting OperationTracer...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.track_head = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1)
            )
            self.seg_head = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1)
            )
        
        def forward(self, x):
            track = self.track_head(x)
            seg = self.seg_head(x)
            return {'track': track, 'seg': seg}
    
    model = TestModel()
    tracer = OperationTracer(model, stage=1, task_heads=['track', 'seg'])
    
    # Test head identification
    tracer._identify_task_heads()
    assert 'track' in tracer.module_to_head.values()
    assert 'seg' in tracer.module_to_head.values()
    
    # Test tracing
    dummy_input = torch.randn(1, 3, 32, 32)
    outputs = tracer.trace(dummy_input)
    
    trace_nodes = tracer.get_trace_data()
    assert len(trace_nodes) > 0, "Should have traced some operations"
    
    # Check that we have nodes from both heads
    task_heads = set(node.task_head for node in trace_nodes if node.task_head)
    assert 'track' in task_heads, "Should have track head operations"
    assert 'seg' in task_heads, "Should have seg head operations"
    
    print(f"✓ OperationTracer tests passed (traced {len(trace_nodes)} operations)")


def test_analyzer():
    """Test TraceAnalyzer functionality"""
    print("\nTesting TraceAnalyzer...")
    
    # Create some dummy trace nodes
    nodes = [
        TraceNode(
            operation="Conv2d",
            module_path="track_head.conv1",
            input_shapes=[(1, 3, 32, 32)],
            output_shapes=[(1, 64, 32, 32)],
            task_head="track",
            memory_usage=1.0,
            compute_time=0.5
        ),
        TraceNode(
            operation="ReLU",
            module_path="track_head.relu",
            input_shapes=[(1, 64, 32, 32)],
            output_shapes=[(1, 64, 32, 32)],
            task_head="track",
            memory_usage=0.5,
            compute_time=0.1
        ),
        TraceNode(
            operation="Conv2d",
            module_path="seg_head.conv1",
            input_shapes=[(1, 3, 32, 32)],
            output_shapes=[(1, 32, 32, 32)],
            task_head="seg",
            memory_usage=0.8,
            compute_time=0.3
        )
    ]
    
    analyzer = TraceAnalyzer()
    analysis = analyzer.analyze(nodes, stage=1)
    
    # Check summary
    assert analysis['summary']['total_operations'] == 3
    assert abs(analysis['summary']['total_memory_mb'] - 2.3) < 0.01
    assert abs(analysis['summary']['total_compute_ms'] - 0.9) < 0.01
    
    # Check head analysis
    head_memory = analysis['head_analysis']['memory_by_head']
    assert abs(head_memory['track'] - 1.5) < 0.01
    assert abs(head_memory['seg'] - 0.8) < 0.01
    
    print("✓ TraceAnalyzer tests passed")


def test_visualizer():
    """Test DataflowVisualizer"""
    print("\nTesting DataflowVisualizer...")
    
    # Create dummy data
    nodes = [
        TraceNode(
            operation="Conv2d",
            module_path="track_head.conv1",
            input_shapes=[(1, 3, 32, 32)],
            output_shapes=[(1, 64, 32, 32)],
            task_head="track",
            memory_usage=10.0
        )
    ]
    
    head_analysis = {
        'memory_by_head': {'track': 10.0, 'seg': 5.0},
        'compute_by_head': {'track': 2.0, 'seg': 1.0}
    }
    
    memory_profile = {
        'total_memory_mb': 15.0,
        'top_consumers': [
            {'operation': 'Conv2d', 'memory_mb': 10.0, 'percentage': 66.7}
        ]
    }
    
    visualizer = DataflowVisualizer(max_nodes=10)
    
    # Test Mermaid generation
    mermaid = visualizer.generate_mermaid(nodes, head_analysis, memory_profile)
    assert '```mermaid' in mermaid
    assert 'graph TB' in mermaid
    assert 'Track Head' in mermaid
    
    # Test memory heatmap
    heatmap = visualizer.generate_memory_heatmap(memory_profile)
    assert 'Memory Usage Heatmap' in heatmap
    assert 'Conv2d' in heatmap
    assert '█' in heatmap  # Progress bar character
    
    print("✓ DataflowVisualizer tests passed")


def test_integration():
    """Test full integration with a model"""
    print("\nTesting full integration...")
    
    # Create a model that mimics UniAD structure
    class MiniUniAD(nn.Module):
        def __init__(self):
            super().__init__()
            # BEV encoder
            self.bev_encoder = nn.Sequential(
                nn.Conv2d(3, 256, 3, padding=1),
                nn.ReLU()
            )
            
            # Task heads
            self.track_head = nn.Conv2d(256, 10, 1)
            self.seg_head = nn.Conv2d(256, 20, 1)
            self.motion_head = nn.Conv2d(256, 30, 1)
            
        def forward(self, x):
            bev = self.bev_encoder(x)
            track = self.track_head(bev)
            seg = self.seg_head(bev)
            motion = self.motion_head(bev)
            
            return {
                'track': track,
                'seg': seg,
                'motion': motion
            }
    
    # Run full pipeline
    model = MiniUniAD()
    tracer = OperationTracer(model, stage=2)
    
    dummy_input = torch.randn(1, 3, 64, 64)
    outputs = tracer.trace(dummy_input)
    
    # Analyze
    trace_nodes = tracer.get_trace_data()
    analyzer = TraceAnalyzer()
    analysis = analyzer.analyze(trace_nodes)
    
    # Visualize
    visualizer = DataflowVisualizer()
    mermaid = visualizer.generate_mermaid(
        trace_nodes,
        analysis['head_analysis'],
        analysis['memory_profile']
    )
    
    # Basic checks
    assert len(trace_nodes) > 0
    assert analysis['summary']['total_operations'] > 0
    assert '```mermaid' in mermaid
    
    # Save test output
    with open('test_output.md', 'w') as f:
        f.write("# Test Output\n\n")
        f.write(f"## Summary\n")
        f.write(f"- Operations: {analysis['summary']['total_operations']}\n")
        f.write(f"- Memory: {analysis['summary']['total_memory_mb']:.2f} MB\n\n")
        f.write("## Visualization\n\n")
        f.write(mermaid)
    
    print("✓ Integration test passed")
    print(f"  - Traced {len(trace_nodes)} operations")
    print(f"  - Total memory: {analysis['summary']['total_memory_mb']:.2f} MB")
    print("  - Output saved to test_output.md")


def main():
    """Run all tests"""
    print("Running PyTorch Operation Tracer Tests")
    print("=" * 50)
    
    try:
        test_tensor_shape_recorder()
        test_trace_node()
        test_operation_tracer()
        test_analyzer()
        test_visualizer()
        test_integration()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()