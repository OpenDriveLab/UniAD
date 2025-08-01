#!/usr/bin/env python3
"""
Example script showing how to use the PyTorch Operation Tracer with UniAD
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pytorch_op_tracer import OperationTracer, TraceAnalyzer, DataflowVisualizer
from pytorch_op_tracer.utils import create_dummy_input


def analyze_trace_results(json_file):
    """Analyze trace results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert nodes to DataFrame
    df = pd.DataFrame(data['nodes'])
    
    # Group by task head
    if 'task_head' in df.columns:
        head_summary = df.groupby('task_head').agg({
            'memory_mb': ['sum', 'mean', 'count'],
            'compute_ms': ['sum', 'mean']
        }).round(2)
        print("\nTask Head Summary:")
        print(head_summary)
    
    # Top memory consumers
    print("\nTop 10 Memory Consumers:")
    top_mem = df.nlargest(10, 'memory_mb')[['operation', 'module_path', 'memory_mb']]
    print(top_mem)
    
    # Operation type summary
    print("\nOperation Type Summary:")
    op_summary = df.groupby('operation').agg({
        'memory_mb': 'sum',
        'compute_ms': 'sum'
    }).round(2).sort_values('memory_mb', ascending=False).head(10)
    print(op_summary)
    
    return df


def visualize_memory_distribution(df):
    """Create visualizations of memory distribution"""
    if df.empty:
        print("No data to visualize")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Memory by task head
    if 'task_head' in df.columns and df['task_head'].notna().any():
        head_memory = df.groupby('task_head')['memory_mb'].sum().sort_values(ascending=True)
        head_memory.plot(kind='barh', ax=ax1)
        ax1.set_xlabel('Memory (MB)')
        ax1.set_title('Memory Usage by Task Head')
    
    # Memory by operation type
    op_memory = df.groupby('operation')['memory_mb'].sum().nlargest(10).sort_values(ascending=True)
    op_memory.plot(kind='barh', ax=ax2)
    ax2.set_xlabel('Memory (MB)')
    ax2.set_title('Top 10 Operations by Memory Usage')
    
    plt.tight_layout()
    plt.savefig('memory_distribution.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to memory_distribution.png")


def example_custom_analysis():
    """Example of custom analysis using the tracer programmatically"""
    import torch
    import torch.nn as nn
    
    # Create a simple model for demonstration
    class SimpleUniADLike(nn.Module):
        def __init__(self):
            super().__init__()
            # Simplified BEV encoder
            self.bev_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 256, 3, padding=1),
                nn.ReLU(),
            )
            
            # Task heads
            self.track_head = nn.Conv2d(256, 10, 1)
            self.seg_head = nn.Conv2d(256, 20, 1)
            self.motion_head = nn.Sequential(
                nn.Conv2d(256 + 10, 128, 3, padding=1),  # Takes track output
                nn.ReLU(),
                nn.Conv2d(128, 30, 1)
            )
            
        def forward(self, x):
            # BEV encoding
            bev_features = self.bev_encoder(x)
            
            # Task heads
            track_out = self.track_head(bev_features)
            seg_out = self.seg_head(bev_features)
            
            # Motion depends on track
            motion_input = torch.cat([bev_features, track_out], dim=1)
            motion_out = self.motion_head(motion_input)
            
            return {
                'track': track_out,
                'seg': seg_out,
                'motion': motion_out
            }
    
    # Create model and tracer
    model = SimpleUniADLike()
    tracer = OperationTracer(model, stage=2, task_heads=['track', 'seg', 'motion'])
    
    # Trace with dummy input
    dummy_input = torch.randn(1, 3, 200, 200)  # Simplified BEV-like input
    outputs = tracer.trace(dummy_input)
    
    # Analyze
    trace_nodes = tracer.get_trace_data()
    analyzer = TraceAnalyzer()
    analysis = analyzer.analyze(trace_nodes)
    
    # Print results
    print("\n=== Custom Model Analysis ===")
    print(f"Total operations: {analysis['summary']['total_operations']}")
    print(f"Total memory: {analysis['summary']['total_memory_mb']:.2f} MB")
    print(f"Total compute: {analysis['summary']['total_compute_ms']:.2f} ms")
    
    # Generate visualization
    visualizer = DataflowVisualizer()
    mermaid = visualizer.generate_mermaid(
        trace_nodes,
        analysis['head_analysis'],
        analysis['memory_profile']
    )
    
    with open('custom_model_trace.md', 'w') as f:
        f.write("# Custom Model Trace\n\n")
        f.write(mermaid)
    
    print("\nSaved custom model trace to custom_model_trace.md")


def main():
    """Main example execution"""
    print("PyTorch Operation Tracer Example")
    print("=" * 50)
    
    # Check if trace results exist
    trace_results = [
        'trace_output.json',
        'stage1_trace.json',
        'stage2_trace.json'
    ]
    
    found_results = [f for f in trace_results if os.path.exists(f)]
    
    if found_results:
        print(f"\nFound existing trace results: {found_results}")
        for result_file in found_results:
            print(f"\nAnalyzing {result_file}...")
            df = analyze_trace_results(result_file)
            
            # Create visualizations
            if not df.empty:
                visualize_memory_distribution(df)
    else:
        print("\nNo existing trace results found.")
        print("Run the tracer first with:")
        print("  python tools/analysis_tools/trace_pytorch_ops.py --config ... --output trace_output.md --export-json")
    
    # Run custom analysis example
    print("\n" + "=" * 50)
    print("Running custom analysis example...")
    example_custom_analysis()


if __name__ == '__main__':
    main()