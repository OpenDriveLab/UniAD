#!/usr/bin/env python3
"""
Main CLI for PyTorch Operation Tracer

Usage:
    python trace_ops.py --config CONFIG_PATH --checkpoint CHECKPOINT_PATH [options]
"""

import argparse
import json
import os
import sys
import time

# Add package to path if running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import OperationTracer
from analyzers import TraceAnalyzer
from visualizers import DataflowVisualizer
from utils import create_dummy_input, load_uniad_model


def main():
    parser = argparse.ArgumentParser(description='PyTorch Operation Tracer for UniAD')
    
    # Model configuration
    parser.add_argument('--config', type=str, help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2], 
                        help='UniAD stage (1: perception, 2: end-to-end)')
    
    # Tracing options
    parser.add_argument('--task-heads', type=str, nargs='+', 
                        default=['track', 'seg', 'motion', 'occ', 'planning'],
                        help='Task heads to trace')
    parser.add_argument('--temporal-frames', type=int, default=3, 
                        help='Number of temporal frames')
    parser.add_argument('--trace-backward', action='store_true', 
                        help='Trace backward pass')
    parser.add_argument('--filter-ops', type=str, nargs='+', 
                        help='Filter specific operations')
    parser.add_argument('--max-nodes', type=int, default=50, 
                        help='Maximum nodes to visualize')
    
    # Analysis options
    parser.add_argument('--memory-profile', action='store_true', 
                        help='Enable memory profiling')
    parser.add_argument('--bev-focus', action='store_true', 
                        help='Focus on BEV operations')
    parser.add_argument('--visualize-temporal', action='store_true', 
                        help='Visualize temporal flow')
    
    # Output options
    parser.add_argument('--output', type=str, default='trace_output.md', 
                        help='Output file path')
    parser.add_argument('--export-json', action='store_true', 
                        help='Export trace data as JSON')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to run on (cuda/cpu)')
    
    # Testing mode
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode with dummy model')
    
    args = parser.parse_args()
    
    # Check config requirement
    if not args.test_mode and not args.config:
        parser.error("--config is required unless --test-mode is specified")
    
    # Initialize model
    if args.test_mode:
        print("Running in test mode with dummy model")
        import torch.nn as nn
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        ).to(args.device)
        cfg = None
    else:
        try:
            print(f"Loading UniAD model from {args.config}")
            model, cfg = load_uniad_model(args.config, args.checkpoint, args.device)
        except ImportError as e:
            print(f"Error: {e}")
            print("Running in fallback mode. Install mmdet3d for full functionality.")
            return 1
    
    # Create tracer
    tracer = OperationTracer(
        model, 
        trace_backward=args.trace_backward,
        filter_ops=args.filter_ops,
        stage=args.stage,
        task_heads=args.task_heads
    )
    
    # Create dummy input
    print("Creating dummy input...")
    dummy_input = create_dummy_input(cfg if not args.test_mode else None, args.device)
    
    # Perform tracing
    print("Tracing model operations...")
    start_time = time.time()
    outputs = tracer.trace(dummy_input)
    trace_time = time.time() - start_time
    
    # Get trace data
    trace_nodes = tracer.get_trace_data()
    print(f"Traced {len(trace_nodes)} operations in {trace_time:.2f} seconds")
    
    # Analyze trace
    print("Analyzing trace data...")
    analyzer = TraceAnalyzer()
    analysis = analyzer.analyze(trace_nodes, stage=args.stage)
    
    # Generate visualizations
    print("Generating visualizations...")
    visualizer = DataflowVisualizer(max_nodes=args.max_nodes)
    mermaid_diagram = visualizer.generate_mermaid(
        trace_nodes, 
        analysis['head_analysis'],
        analysis['memory_profile']
    )
    memory_heatmap = visualizer.generate_memory_heatmap(analysis['memory_profile'])
    
    # Write output
    print(f"Writing output to {args.output}")
    with open(args.output, 'w') as f:
        f.write(f"# UniAD PyTorch Operation Trace Report\n\n")
        f.write(f"**Model**: {args.config}\n")
        f.write(f"**Stage**: {args.stage}\n")
        f.write(f"**Trace Time**: {trace_time:.2f} seconds\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        summary = analysis['summary']
        f.write(f"- Total Operations: {summary['total_operations']}\n")
        f.write(f"- Unique Operations: {summary['unique_operations']}\n")
        f.write(f"- Total Memory: {summary['total_memory_mb']:.1f} MB\n")
        f.write(f"- Total Compute Time: {summary['total_compute_ms']:.1f} ms\n\n")
        
        # Task Head Analysis
        f.write("## Task Head Analysis\n\n")
        head_memory = analysis['head_analysis']['memory_by_head']
        head_compute = analysis['head_analysis']['compute_by_head']
        for head in ['track', 'seg', 'motion', 'occ', 'planning']:
            if head in head_memory:
                f.write(f"- **{head.capitalize()}**: {head_memory[head]:.1f} MB, {head_compute[head]:.1f} ms\n")
        f.write("\n")
        
        # BEV Analysis
        if args.bev_focus:
            f.write("## BEV Analysis\n\n")
            bev = analysis['bev_analysis']
            f.write(f"- Total BEV Operations: {bev['total_bev_ops']}\n")
            f.write(f"- BEV Memory: {bev['bev_memory_mb']:.1f} MB\n")
            f.write(f"- Grid Sizes: {bev['grid_sizes']}\n")
            f.write(f"- Frozen Encoder: {bev['frozen_encoder']}\n\n")
        
        # Dataflow Diagram
        f.write("## Dataflow Visualization\n\n")
        f.write(mermaid_diagram)
        f.write("\n\n")
        
        # Memory Heatmap
        if args.memory_profile:
            f.write("## Memory Profile\n\n")
            f.write(memory_heatmap)
            f.write("\n\n")
    
    # Export JSON if requested
    if args.export_json:
        json_path = args.output.replace('.md', '.json')
        print(f"Exporting trace data to {json_path}")
        trace_data = {
            'config': args.config,
            'stage': args.stage,
            'analysis': analysis,
            'nodes': [node.to_dict() for node in trace_nodes]
        }
        with open(json_path, 'w') as f:
            json.dump(trace_data, f, indent=2)
    
    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())