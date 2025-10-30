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

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List

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


@dataclass
class BenchmarkConfig:
    """Configuration for a single AllReduce benchmark run"""
    # MPI and device info
    world_size: int
    rank: int
    local_rank: int
    gpus_per_node: int
    mapping: Mapping

    # Data type
    torch_dtype: torch.dtype
    dtype_name: str
    dtype_size_bytes: int

    # Tensor dimensions
    hidden_size: int  # Fixed hidden dimension (e.g., 4096)
    min_tokens: int  # Minimum number of tokens
    max_tokens: int  # Maximum number of tokens
    ratio: int  # Multiplicative ratio for token count iteration

    # Single strategy and operation for this config
    strategy: AllReduceStrategy
    fusion_op: AllReduceFusionOp

    # Benchmark parameters
    inner_loop: int = 1200
    outer_loop: int = 10
    enable_cudagraph: bool = False

    # Output control
    print_header: bool = True


@dataclass
class EnvironmentSetup:
    """Shared environment setup that's common across all benchmark configs"""
    world_size: int
    rank: int
    local_rank: int
    gpus_per_node: int
    mapping: Mapping
    torch_dtype: torch.dtype
    dtype_name: str
    dtype_size_bytes: int
    hidden_size: int
    min_tokens: int
    max_tokens: int
    ratio: int


def initialize_environment(dtype: str, token_range: str, hidden_size: int,
                           only_ub: bool) -> EnvironmentSetup:
    """
    Initialize MPI environment, devices, and user buffers.
    This is called once at the start and shared across all benchmark configs.

    Args:
        dtype: Data type string (e.g., "float16", "bfloat16")
        token_range: Token count range string "min_tokens,max_tokens,ratio"
        hidden_size: Fixed hidden dimension size
        only_ub: Whether to only test user buffer strategy

    Returns:
        EnvironmentSetup with initialized parameters
    """
    tllm.logger.set_level("error")

    # MPI setup
    world_size = tllm.mpi_world_size()
    rank = tllm.mpi_rank()
    local_rank = local_mpi_rank()
    gpus_per_node = local_mpi_size()

    if world_size == 1:
        raise RuntimeError("Benchmark must run with mpi_world_size > 1")

    # Device setup
    torch.cuda.set_device(local_rank)
    cudart.cudaSetDevice(local_rank)

    mapping = Mapping(world_size, rank, gpus_per_node, tp_size=world_size)

    # Data type setup
    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
    dtype_size_bytes = torch_dtype.itemsize

    # Parse token range
    min_tokens, max_tokens, ratio = [int(i) for i in token_range.split(",")]

    # Calculate max bytes needed for user buffers
    max_elements = max_tokens * hidden_size
    max_bytes = max_elements * dtype_size_bytes

    # Initialize user buffers (needed for all strategies)
    ub.initialize_userbuffers_manager(
        world_size,
        1,
        1,
        rank,
        torch.cuda.device_count(),
        max_bytes,
        not only_ub  # use_multicast for non-UB strategies
    )

    return EnvironmentSetup(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        gpus_per_node=gpus_per_node,
        mapping=mapping,
        torch_dtype=torch_dtype,
        dtype_name=dtype,
        dtype_size_bytes=dtype_size_bytes,
        hidden_size=hidden_size,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        ratio=ratio,
    )


def create_benchmark_config(
    env: EnvironmentSetup,
    strategy: AllReduceStrategy,
    fusion_op: AllReduceFusionOp,
    enable_cudagraph: bool = False,
    inner_loop: int = 1200,
    outer_loop: int = 10,
    print_header: bool = True,
) -> BenchmarkConfig:
    """
    Create a benchmark configuration for a specific strategy and operation.

    Args:
        env: Environment setup (shared across all configs)
        strategy: AllReduce strategy to test
        fusion_op: Fusion operation to test
        enable_cudagraph: Whether to enable CUDA graph
        inner_loop: Number of inner loop iterations
        outer_loop: Number of outer loop iterations for timing
        print_header: Whether to print header

    Returns:
        BenchmarkConfig for this specific strategy and operation
    """
    return BenchmarkConfig(
        world_size=env.world_size,
        rank=env.rank,
        local_rank=env.local_rank,
        gpus_per_node=env.gpus_per_node,
        mapping=env.mapping,
        torch_dtype=env.torch_dtype,
        dtype_name=env.dtype_name,
        dtype_size_bytes=env.dtype_size_bytes,
        hidden_size=env.hidden_size,
        min_tokens=env.min_tokens,
        max_tokens=env.max_tokens,
        ratio=env.ratio,
        strategy=strategy,
        fusion_op=fusion_op,
        inner_loop=inner_loop,
        outer_loop=outer_loop,
        enable_cudagraph=enable_cudagraph,
        print_header=print_header,
    )


def execute_benchmark(config: BenchmarkConfig) -> None:
    """
    Execute benchmark for the given configuration across all sizes.

    Args:
        config: Benchmark configuration with strategy, fusion_op, and size range
    """
    # Print header if this is the first benchmark
    if config.mapping.rank == 0 and config.print_header:
        print(f"{'world_size':<15}, "
              f"{'dtype':<10}, "
              f"{'num_tokens':<12}, "
              f"{'hidden_size':<12}, "
              f"{'tensor size (B)':<16}, "
              f"{'strategy':<15}, "
              f"{'fusion':<25}, "
              f"{'duration (ms)':<12}")

    # Create allreduce module with dtype for MNNVL support
    allreduce = AllReduce(mapping=config.mapping,
                          strategy=config.strategy,
                          dtype=config.torch_dtype)

    # Iterate over token counts
    num_tokens = config.min_tokens
    while num_tokens <= config.max_tokens:
        # Skip large fusion ops for very large token counts
        total_elements = num_tokens * config.hidden_size
        if total_elements >= 25600000 and config.fusion_op != AllReduceFusionOp.NONE:
            num_tokens *= config.ratio
            continue

        # Create input tensor with shape (num_tokens, hidden_size)
        input_tensor = torch.ones((num_tokens, config.hidden_size),
                                  dtype=config.torch_dtype,
                                  device="cuda")

        # Setup parameters based on fusion operation
        if config.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
            norm_weight = torch.randn((config.hidden_size, ),
                                      dtype=config.torch_dtype,
                                      device="cuda")
            norm = RMSNorm(hidden_size=config.hidden_size,
                           dtype=config.torch_dtype,
                           eps=1e-5).cuda()
            norm.weight.data.copy_(norm_weight)

            params = {
                "all_reduce_params":
                AllReduceParams(
                    fusion_op=config.fusion_op,
                    residual=input_tensor,
                    norm_weight=norm.weight,
                    eps=norm.variance_epsilon,
                )
            }
        else:
            params = {
                "all_reduce_params": AllReduceParams(fusion_op=config.fusion_op)
            }

        # Define benchmark function
        def func(input):
            for _ in range(config.inner_loop):
                input = allreduce(input, **params)
                if config.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
                    input = input[0]
            return input

        # Setup timing events
        start_events = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(config.outer_loop)
        ]
        stop_events = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(config.outer_loop)
        ]

        # Run benchmark
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            if config.enable_cudagraph:
                # Warmup for CUDA graph
                graph = torch.cuda.CUDAGraph()
                for _ in range(2):
                    func(input_tensor)
                with torch.cuda.graph(graph, stream=stream):
                    output = func(input_tensor)

            tllm.mpi_barrier()
            delay_kernel(2000000, stream)
            torch.cuda.profiler.start()

            for i in range(config.outer_loop):
                start_events[i].record(stream)
                if config.enable_cudagraph:
                    graph.replay()
                else:
                    output = func(input_tensor)
                stop_events[i].record(stream)

        torch.cuda.synchronize()
        torch.cuda.profiler.stop()

        # Calculate median runtime
        runtimes = [
            start_events[i].elapsed_time(stop_events[i])
            for i in range(config.outer_loop)
        ]
        median_ms = sorted(runtimes)[len(runtimes) // 2] / config.inner_loop

        # Verify correctness for NONE fusion
        if config.fusion_op == AllReduceFusionOp.NONE:
            allreduce_ref = (input_tensor *
                             config.world_size)**config.inner_loop
            torch.testing.assert_close(output, allreduce_ref)

        # Print results from rank 0
        if config.mapping.rank == 0:
            total_elements = num_tokens * config.hidden_size
            message_size_b = total_elements * config.dtype_size_bytes
            print(f"{config.world_size:<15}, "
                  f"{config.dtype_name:<10}, "
                  f"{num_tokens:<12}, "
                  f"{config.hidden_size:<12}, "
                  f"{message_size_b:<16.0f}, "
                  f"{config.strategy.name:<15}, "
                  f"{config.fusion_op.name:<25}, "
                  f"{median_ms:<12.5f}")

        # Update token count for next iteration
        num_tokens *= config.ratio


def get_default_strategies(only_ub: bool) -> List[AllReduceStrategy]:
    """Get the default list of strategies to test"""
    if only_ub:
        return [AllReduceStrategy.UB]
    else:
        return [
            AllReduceStrategy.NCCL,
            AllReduceStrategy.MIN_LATENCY,
            AllReduceStrategy.NCCL_SYMMETRIC,
            AllReduceStrategy.NCCL_DEVICE,
            AllReduceStrategy.MNNVL,
        ]


def get_default_fusion_ops(only_ub: bool) -> List[AllReduceFusionOp]:
    """Get the default list of fusion operations to test"""
    if only_ub:
        return [AllReduceFusionOp.RESIDUAL_RMS_NORM]
    else:
        return [
            AllReduceFusionOp.NONE,
            AllReduceFusionOp.RESIDUAL_RMS_NORM,
        ]


def launch_benchmarks(
    env: EnvironmentSetup,
    strategies: List[AllReduceStrategy],
    fusion_ops: List[AllReduceFusionOp],
    enable_cudagraph: bool = False,
    inner_loop: int = 1200,
    outer_loop: int = 10,
) -> None:
    """
    Launch benchmarks for all strategy and fusion operation combinations.
    Creates a separate config for each combination and executes them sequentially.

    Args:
        env: Environment setup (shared)
        strategies: List of strategies to test
        fusion_ops: List of fusion operations to test
        enable_cudagraph: Whether to enable CUDA graph
        inner_loop: Number of inner loop iterations
        outer_loop: Number of outer loop iterations
    """
    # Track whether we've printed the header
    first_run = True

    # Create and execute a config for each combination
    for strategy in strategies:
        for fusion_op in fusion_ops:
            config = create_benchmark_config(
                env=env,
                strategy=strategy,
                fusion_op=fusion_op,
                enable_cudagraph=enable_cudagraph,
                inner_loop=inner_loop,
                outer_loop=outer_loop,
                print_header=first_run,
            )
            execute_benchmark(config)
            first_run = False


def main():
    """Main entry point for the benchmark"""
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
    parser.add_argument("--only-ub",
                        action="store_true",
                        help="Only test user buffer strategy")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Test only specific strategy (e.g., MNNVL, NCCL_DEVICE)")
    parser.add_argument(
        "--fusion-op",
        type=str,
        default=None,
        help="Test only specific fusion op (e.g., NONE, RESIDUAL_RMS_NORM)")
    parser.add_argument("--inner-loop",
                        type=int,
                        default=1200,
                        help="Number of inner loop iterations")
    parser.add_argument("--outer-loop",
                        type=int,
                        default=10,
                        help="Number of outer loop iterations for timing")
    args = parser.parse_args()

    # Initialize environment (once, shared across all configs)
    env = initialize_environment(args.dtype, args.token_range, args.hidden_size,
                                 args.only_ub)

    # Determine which strategies to test
    if args.strategy:
        strategy_map = {
            "NCCL": AllReduceStrategy.NCCL,
            "MIN_LATENCY": AllReduceStrategy.MIN_LATENCY,
            "NCCL_SYMMETRIC": AllReduceStrategy.NCCL_SYMMETRIC,
            "NCCL_DEVICE": AllReduceStrategy.NCCL_DEVICE,
            "MNNVL": AllReduceStrategy.MNNVL,
            "UB": AllReduceStrategy.UB,
        }
        if args.strategy.upper() not in strategy_map:
            raise ValueError(f"Unknown strategy: {args.strategy}")
        strategies = [strategy_map[args.strategy.upper()]]
    else:
        strategies = get_default_strategies(args.only_ub)

    # Determine which fusion ops to test
    if args.fusion_op:
        fusion_map = {
            "NONE": AllReduceFusionOp.NONE,
            "RESIDUAL_RMS_NORM": AllReduceFusionOp.RESIDUAL_RMS_NORM,
        }
        if args.fusion_op.upper() not in fusion_map:
            raise ValueError(f"Unknown fusion op: {args.fusion_op}")
        fusion_ops = [fusion_map[args.fusion_op.upper()]]
    else:
        fusion_ops = get_default_fusion_ops(args.only_ub)

    # Launch benchmarks (creates separate config for each combination)
    launch_benchmarks(
        env=env,
        strategies=strategies,
        fusion_ops=fusion_ops,
        enable_cudagraph=args.enable_cudagraph,
        inner_loop=args.inner_loop,
        outer_loop=args.outer_loop,
    )


if __name__ == "__main__":
    main()
