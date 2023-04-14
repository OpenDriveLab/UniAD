# The throughput is benchmarked with the max pow-of-2 batch size
import argparse
import torch
import time
import os
import json
import logging
import csv
from torch.utils.tensorboard import SummaryWriter
import sys

# sys.path.append("../")
from Profile.models import build_model

# from models import build_model

MB = 1024.0 * 1024.0


def print_and_log(s):
    print(s)
    logging.info(s)


def get_args_parser():
    parser = argparse.ArgumentParser('Throughput in GPU measurement script', add_help=False)
    # Model Configuration
    parser.add_argument('--model', default=None, type=str,
                        help='the type of the model to benchmark')
    parser.add_argument('--seq_len', default=None, type=int,
                        help='the seq_len of the model')
    parser.add_argument('--dim', default=None, type=int,
                        help='the dim of the model')
    parser.add_argument('--heads', default=None, type=int,
                        help='the number of heads of the model')
    # Profile Configuration
    parser.add_argument('--warmup_iters', default=3, type=int,
                        help='the iterations used for warm up')
    parser.add_argument('--num_iters', default=30, type=int,
                        help='the iterations used for benchmark')
    parser.add_argument('--init_batch_size', default=4096, type=int,
                        help='the init batch size')
    parser.add_argument('--enable_op_profiling', action='store_true', default=False, help='Profiling each operator')
    return parser


'''
这段代码定义了一个名为 compute_throughput 的函数，用于基于 PyTorch 模型计算 GPU 上的吞吐量和最大内存使用量。

函数接受以下参数：

    model_name：模型名称。
    model_config：模型配置字典，包括序列长度、特征维度等信息。
    batch_size：批次大小。
    warmup_iters：热身迭代次数，默认值为 3。
    num_iters：测试迭代次数，默认值为 30。
    enable_op_profiling：是否启用操作分析。默认为 False。

函数首先清空 GPU 缓存，并将模型加载到 GPU 设备上。然后，根据输入大小生成输入张量。如果启用了操作分析，
则使用 PyTorch Profiler 进行性能分析；否则，使用 Python 时间模块测量模型推理时间。
最后，函数返回每秒处理的图像数（imgs_per_sec）以及最大内存使用量。

在启用操作分析时，函数使用 PyTorch Profiler 对模型进行分析，并将结果保存到 TensorBoard 日志中。
最后，函数解析操作分析结果并计算每秒处理的图像数（imgs_per_sec）。

在未启用操作分析时，函数使用 Python 时间模块测量推理时间。函数首先执行多次热身迭代，然后执行多次测试迭代，
并记录每个迭代的推理时间。函数使用这些数据计算平均推理时间，并计算每秒处理的图像数（imgs_per_sec）。

最后，函数打印并记录每个模型配置和批次大小的吞吐量和最大内存使用量。
'''


@torch.no_grad()
def compute_throughput(model_name, model_config, batch_size, warmup_iters=3, num_iters=30, enable_op_profiling=False):
    torch.cuda.empty_cache()
    device = torch.device('cuda')

    # TODO replace ViT model with UniAd
    # model is derived from nn.Module
    model = build_model(model_name, model_config)
    model.eval()
    model.to(device)
    timing = []

    # TODO Identify UniAD input
    # Here each inputs work as one iteration, need to find how to group UniAD's dataset as input
    inputs = torch.randn(batch_size, model_config["seq_len"], model_config["dim"], device=device)

    # Collect performance metrics like flops
    if enable_op_profiling:
        tfb_writer = SummaryWriter(os.path.join("benchmark_logs/{}".format(model_name)))
        tfb_writer.add_graph(model, inputs)
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=warmup_iters,
                    active=num_iters,
                    repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join("benchmark_logs/{}".format(model_name))),
                with_stack=False,
                # with_stack=True incurs an additional overhead, and is better suited for investigating code. Remember to remove it if you are benchmarking performance.
                profile_memory=True,
                with_flops=True
        ) as profiler:
            for step in range(2 + warmup_iters + num_iters):
                model(inputs)
                torch.cuda.synchronize()
                profiler.step()
        key_averages_by_cuda_time = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        # save to the log
        logging.info(key_averages_by_cuda_time)
        time_per_image = key_averages_by_cuda_time.split("\n")[-2].split(" ")[-1][:-1]
        if "m" in time_per_image:
            time_per_image = float(time_per_image[:-1]) / 1000
        else:
            time_per_image = float(time_per_image)
        imgs_per_sec = batch_size * num_iters / time_per_image

    # Or collect inference latency
    else:
        # warmup
        for _ in range(warmup_iters):
            model(inputs)

        torch.cuda.synchronize()
        for _ in range(num_iters):
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
        timing = torch.as_tensor(timing, dtype=torch.float32)
        imgs_per_sec = batch_size / timing.mean()

    memory = torch.cuda.max_memory_allocated() / MB

    print_and_log(f"====> {model_config} (batch_size: {batch_size}): {imgs_per_sec:.2f} FPS, max mem: {memory:.2f} MB")
    return imgs_per_sec


def locate_arch_T(model, model_config, init_batch_size, warmup_iters=3, num_iters=30, enable_op_profiling=False,
                  return_bs=False):
    assert (init_batch_size & (init_batch_size - 1) == 0) and init_batch_size != 0, (
        "batch_size_step should be power-of-2")
    batch_size = init_batch_size

    # imgs_per_sec = compute_throughput(model, model_config, batch_size, warmup_iters=warmup_iters,
    # num_iters=num_iters, enable_op_profiling=enable_op_profiling)

    LOOP_FLAG = True
    while LOOP_FLAG and batch_size != 0:
        LOOP_FLAG = False
        try:
            imgs_per_sec = compute_throughput(model, model_config, batch_size, warmup_iters=warmup_iters,
                                              num_iters=num_iters, enable_op_profiling=enable_op_profiling)
        except:
            if batch_size == 1:
                batch_size = 0
            else:
                batch_size = max(int(batch_size / 2), 1)
            print_and_log("{} - Down scale batch_size to {}".format(model_config, batch_size))
            LOOP_FLAG = True
    if return_bs:
        return imgs_per_sec, batch_size
    else:
        return imgs_per_sec


def main(args):
    model_name = "{}-seq_len_{}-dim_{}-heads_{}".format(args.model, args.seq_len, args.dim, args.heads)

    basic_arch = {
        "seq_len": args.seq_len,
        "dim": args.dim,
        "heads": args.heads,
    }

    os.makedirs("benchmark_logs", exist_ok=True)

    logging.basicConfig(filename=os.path.join("benchmark_logs", "{}.log".format(model_name)), level=logging.DEBUG)
    locate_arch_T(model_name,
                  basic_arch,
                  args.init_batch_size,
                  warmup_iters=args.warmup_iters,
                  num_iters=args.num_iters,
                  enable_op_profiling=args.enable_op_profiling)


def benchmark():
    parser = argparse.ArgumentParser('Throughput in GPU measurement script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    
