# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from argparse import ArgumentParser

# isort: off
import torch

# isort: on
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

import tensorrt_llm as tllm
import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm import Mapping
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import local_mpi_rank, local_mpi_size
from tensorrt_llm.bindings.internal.runtime import delay_kernel
from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy


def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(description="AllReduce microbenchmark")
    parser.add_argument("--dtype",
                        "-t",
                        default="float16",
                        help="Data type (float16, bfloat16, float32)")
    parser.add_argument(
        "--token-range",
        "-r",
        default="1,65536,4",
        help="Token count range: min_tokens,max_tokens,multiplicative_ratio")
    parser.add_argument("--hidden-size",
                        type=int,
                        default=4096,
                        help="Hidden dimension size (default: 4096)")
    parser.add_argument("--enable-cudagraph",
                        action="store_true",
                        help="Enable CUDA graph capture")
    parser.add_argument(
        "--strategy",
        type=str,
        default="MNNVL",
        help=
        "Strategy to test: NCCL, MIN_LATENCY, NCCL_SYMMETRIC, NCCL_DEVICE, MNNVL, UB (default: MNNVL)"
    )
    parser.add_argument(
        "--fusion-op",
        type=str,
        nargs='*',
        default=None,
        help=
        "Fusion ops to test (default: NONE RESIDUAL_RMS_NORM). Example: --fusion-op NONE RESIDUAL_RMS_NORM"
    )
    parser.add_argument("--inner-loop",
                        type=int,
                        default=1200,
                        help="Number of inner loop iterations")
    parser.add_argument("--outer-loop",
                        type=int,
                        default=10,
                        help="Number of outer loop iterations for timing")
    args = parser.parse_args()

    # Validate and convert strategy
    strategy_map = {
        "NCCL": AllReduceStrategy.NCCL,
        "MIN_LATENCY": AllReduceStrategy.MIN_LATENCY,
        "NCCL_SYMMETRIC": AllReduceStrategy.NCCL_SYMMETRIC,
        "NCCL_DEVICE": AllReduceStrategy.NCCL_DEVICE,
        "MNNVL": AllReduceStrategy.MNNVL,
        "UB": AllReduceStrategy.UB,
    }
    strategy_name = args.strategy.strip().upper()
    if strategy_name not in strategy_map:
        parser.error(f"Unknown strategy: {args.strategy}. "
                     f"Valid options: {', '.join(strategy_map.keys())}")
    args.strategy = strategy_map[strategy_name]

    # Validate and convert fusion ops
    if args.fusion_op:
        fusion_map = {
            "NONE": AllReduceFusionOp.NONE,
            "RESIDUAL_RMS_NORM": AllReduceFusionOp.RESIDUAL_RMS_NORM,
        }
        fusion_ops = []
        for arg in args.fusion_op:
            for name in arg.split(','):
                name = name.strip().upper()
                if name not in fusion_map:
                    parser.error(
                        f"Unknown fusion op: {name}. "
                        f"Valid options: {', '.join(fusion_map.keys())}")
                fusion_ops.append(fusion_map[name])
        args.fusion_op = fusion_ops
    else:
        # Default: test both NONE and RESIDUAL_RMS_NORM
        args.fusion_op = [
            AllReduceFusionOp.NONE, AllReduceFusionOp.RESIDUAL_RMS_NORM
        ]

    # Parse token range
    try:
        min_tokens, max_tokens, ratio = [
            int(i) for i in args.token_range.split(",")
        ]
        if min_tokens <= 0 or max_tokens <= 0 or ratio <= 0:
            parser.error("Token range values must be positive")
        if min_tokens > max_tokens:
            parser.error("min_tokens must be <= max_tokens")
        args.min_tokens = min_tokens
        args.max_tokens = max_tokens
        args.ratio = ratio
    except ValueError:
        parser.error(
            "Invalid token-range format. Expected: min,max,ratio (e.g., 1,65536,4)"
        )

    return args


def run_benchmark(args):
    """
    Run the AllReduce benchmark with the given arguments.

    Args:
        args: Parsed and validated arguments with:
            - strategy: AllReduceStrategy enum
            - fusion_op: List of AllReduceFusionOp enums
            - min_tokens, max_tokens, ratio: Parsed integers
            - dtype, hidden_size, enable_cudagraph, inner_loop, outer_loop
    """
    # MPI setup
    world_size = tllm.mpi_world_size()
    rank = tllm.mpi_rank()
    local_rank = local_mpi_rank()
    gpus_per_node = local_mpi_size()

    if world_size == 1:
        if rank == 0:
            print("ERROR: Benchmark must run with mpi_world_size > 1",
                  file=sys.stderr,
                  flush=True)
        sys.exit(1)

    # Device setup
    torch.cuda.set_device(local_rank)
    cudart.cudaSetDevice(local_rank)
    mapping = Mapping(world_size, rank, gpus_per_node, tp_size=world_size)

    # Data type setup
    torch_dtype = tllm._utils.str_dtype_to_torch(args.dtype)
    dtype_size_bytes = torch_dtype.itemsize

    # Extract validated values
    strategy = args.strategy
    fusion_ops = args.fusion_op
    min_tokens = args.min_tokens
    max_tokens = args.max_tokens
    ratio = args.ratio

    # Initialize user buffers based on strategy
    max_elements = args.max_tokens * args.hidden_size
    max_bytes = max_elements * dtype_size_bytes

    if args.strategy in [
            AllReduceStrategy.NCCL_SYMMETRIC, AllReduceStrategy.NCCL_DEVICE
    ]:
        # These need UB with use_multicast=True
        ub.initialize_userbuffers_manager(world_size, 1, 1, rank,
                                          torch.cuda.device_count(), max_bytes,
                                          True)
    elif args.strategy == AllReduceStrategy.UB:
        # UB needs use_multicast=False
        ub.initialize_userbuffers_manager(world_size, 1, 1, rank,
                                          torch.cuda.device_count(), max_bytes,
                                          False)
    # NCCL, MIN_LATENCY, MNNVL don't need UB initialization

    allreduce = AllReduce(mapping=mapping, strategy=strategy, dtype=torch_dtype)

    # Run a test operation to verify strategy works before benchmarking
    if rank == 0:
        print(f"\nTesting {strategy.name} with a small operation...",
              flush=True)

    test_tensor = torch.ones((128, args.hidden_size),
                             dtype=torch_dtype,
                             device="cuda")

    # Use appropriate fusion op for testing (UB doesn't support NONE)
    test_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM if strategy == AllReduceStrategy.UB else AllReduceFusionOp.NONE

    if test_fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
        # Setup RMS norm for testing
        norm_weight = torch.randn((args.hidden_size, ),
                                  dtype=torch_dtype,
                                  device="cuda")
        norm = RMSNorm(hidden_size=args.hidden_size,
                       dtype=torch_dtype,
                       eps=1e-5).cuda()
        norm.weight.data.copy_(norm_weight)
        test_params = AllReduceParams(
            fusion_op=test_fusion_op,
            residual=test_tensor,
            norm_weight=norm.weight,
            eps=norm.variance_epsilon,
        )
    else:
        test_params = AllReduceParams(fusion_op=test_fusion_op)

    try:
        test_output = allreduce(test_tensor, all_reduce_params=test_params)
        torch.cuda.synchronize()

        # Verify correctness for NONE fusion
        if test_fusion_op == AllReduceFusionOp.NONE:
            expected = test_tensor * world_size
            torch.testing.assert_close(test_output, expected)

        if rank == 0:
            print(
                f"Test passed! Strategy {strategy.name} is working correctly.",
                flush=True)
    except Exception as e:
        if rank == 0:
            print(
                f"\nERROR: Test operation failed for strategy {strategy.name}!",
                file=sys.stderr,
                flush=True)
            print(f"Error: {e}", file=sys.stderr, flush=True)

        # Ensure all ranks exit together
        tllm.mpi_barrier()
        sys.exit(1)

    # Print header
    if rank == 0:
        print(f"{'world_size':<15}, "
              f"{'dtype':<10}, "
              f"{'num_tokens':<12}, "
              f"{'hidden_size':<12}, "
              f"{'tensor size (B)':<16}, "
              f"{'strategy':<15}, "
              f"{'fusion':<25}, "
              f"{'duration (ms)':<12}")

    # Benchmark each fusion operation
    for fusion_op in fusion_ops:
        # Skip NONE fusion for UB strategy (UB doesn't support NONE)
        if strategy == AllReduceStrategy.UB and fusion_op == AllReduceFusionOp.NONE:
            if rank == 0:
                print(
                    f"Skipping {strategy.name} with {fusion_op.name} (not supported)",
                    flush=True)
            continue

        # Iterate over token counts
        num_tokens = min_tokens
        while num_tokens <= max_tokens:
            # Skip large fusion ops for very large token counts
            total_elements = num_tokens * args.hidden_size
            if total_elements >= 25600000 and fusion_op != AllReduceFusionOp.NONE:
                num_tokens *= ratio
                continue

            # Create input tensor
            input_tensor = torch.ones((num_tokens, args.hidden_size),
                                      dtype=torch_dtype,
                                      device="cuda")

            # Setup parameters based on fusion operation
            if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
                norm_weight = torch.randn((args.hidden_size, ),
                                          dtype=torch_dtype,
                                          device="cuda")
                norm = RMSNorm(hidden_size=args.hidden_size,
                               dtype=torch_dtype,
                               eps=1e-5).cuda()
                norm.weight.data.copy_(norm_weight)

                params = AllReduceParams(
                    fusion_op=fusion_op,
                    residual=input_tensor,
                    norm_weight=norm.weight,
                    eps=norm.variance_epsilon,
                )
            else:
                params = AllReduceParams(fusion_op=fusion_op)

            # Define benchmark function
            def func(input):
                for _ in range(args.inner_loop):
                    input = allreduce(input, all_reduce_params=params)
                    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
                        input = input[0]
                return input

            # Setup timing events
            start_events = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(args.outer_loop)
            ]
            stop_events = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(args.outer_loop)
            ]

            # Run benchmark
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                if args.enable_cudagraph:
                    # Warmup for CUDA graph
                    graph = torch.cuda.CUDAGraph()
                    for _ in range(2):
                        func(input_tensor)
                    with torch.cuda.graph(graph, stream=stream):
                        output = func(input_tensor)

                tllm.mpi_barrier()
                delay_kernel(2000000, stream)
                torch.cuda.profiler.start()

                for i in range(args.outer_loop):
                    start_events[i].record(stream)
                    if args.enable_cudagraph:
                        graph.replay()
                    else:
                        output = func(input_tensor)
                    stop_events[i].record(stream)

            torch.cuda.synchronize()
            torch.cuda.profiler.stop()

            # Calculate median runtime
            runtimes = [
                start_events[i].elapsed_time(stop_events[i])
                for i in range(args.outer_loop)
            ]
            median_ms = sorted(runtimes)[len(runtimes) // 2] / args.inner_loop

            # Verify correctness for NONE fusion
            if fusion_op == AllReduceFusionOp.NONE:
                allreduce_ref = (input_tensor * world_size)**args.inner_loop
                torch.testing.assert_close(output, allreduce_ref)

            # Print results from rank 0
            if rank == 0:
                message_size_b = total_elements * dtype_size_bytes
                print(f"{world_size:<15}, "
                      f"{args.dtype:<10}, "
                      f"{num_tokens:<12}, "
                      f"{args.hidden_size:<12}, "
                      f"{message_size_b:<16.0f}, "
                      f"{strategy.name:<15}, "
                      f"{fusion_op.name:<25}, "
                      f"{median_ms:<12.5f}")

            # Update token count for next iteration
            num_tokens *= ratio


def main():
    """Main entry point"""
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
