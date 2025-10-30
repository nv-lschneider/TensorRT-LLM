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

import contextlib
import io
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Optional

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
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
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


def initialize_environment(dtype: str, token_range: str,
                           hidden_size: int) -> EnvironmentSetup:
    """
    Initialize MPI environment and devices (but NOT user buffers).
    User buffers are initialized separately based on which strategies are tested.

    Args:
        dtype: Data type string (e.g., "float16", "bfloat16")
        token_range: Token count range string "min_tokens,max_tokens,ratio"
        hidden_size: Fixed hidden dimension size

    Returns:
        EnvironmentSetup with initialized parameters
    """
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


def initialize_userbuffers_for_strategies(
        env: EnvironmentSetup, strategies: List[AllReduceStrategy]) -> None:
    """
    Initialize user buffers based on which strategies are being tested.

    Different strategies have different UB requirements:
    - NCCL_SYMMETRIC, NCCL_DEVICE: need UB with use_multicast=True
    - UB: needs UB with use_multicast=False
    - NCCL, MIN_LATENCY, MNNVL: don't need UB initialization

    Args:
        env: Environment setup with device and size info
        strategies: List of strategies that will be tested
    """
    # Calculate max bytes needed
    max_elements = env.max_tokens * env.hidden_size
    max_bytes = max_elements * env.dtype_size_bytes

    # Check which strategies need UB
    needs_ub_with_multicast = any(
        s in [AllReduceStrategy.NCCL_SYMMETRIC, AllReduceStrategy.NCCL_DEVICE]
        for s in strategies)
    needs_ub_without_multicast = AllReduceStrategy.UB in strategies

    if needs_ub_with_multicast and needs_ub_without_multicast:
        # Both types needed - initialize with multicast=True (more general)
        # Note: This is a limitation - can't test both types in same run
        if env.rank == 0:
            print(
                "Warning: Testing both UB and NCCL_SYMMETRIC/NCCL_DEVICE together. "
                "Initializing with use_multicast=True.")
        ub.initialize_userbuffers_manager(env.world_size, 1, 1, env.rank,
                                          torch.cuda.device_count(), max_bytes,
                                          True)
    elif needs_ub_with_multicast:
        # NCCL_SYMMETRIC or NCCL_DEVICE
        ub.initialize_userbuffers_manager(env.world_size, 1, 1, env.rank,
                                          torch.cuda.device_count(), max_bytes,
                                          True)
    elif needs_ub_without_multicast:
        # UB strategy
        ub.initialize_userbuffers_manager(env.world_size, 1, 1, env.rank,
                                          torch.cuda.device_count(), max_bytes,
                                          False)
    # else: No UB initialization needed


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


def execute_benchmark(config: BenchmarkConfig) -> bool:
    """
    Execute benchmark for the given configuration across all sizes.

    Args:
        config: Benchmark configuration with strategy, fusion_op, and size range

    Returns:
        True if benchmark completed successfully, False if it should stop iteration
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
    # Capture logs to detect fallback
    try:
        with capture_trtllm_logs() as log_lines:
            allreduce = AllReduce(mapping=config.mapping,
                                  strategy=config.strategy,
                                  dtype=config.torch_dtype)

            # Check if initialization caused a fallback
            fallback_msg = check_for_strategy_fallback(log_lines,
                                                       config.strategy)
            if fallback_msg:
                if config.mapping.rank == 0:
                    print(f"Skipping {config.strategy.name}: {fallback_msg}")
                return False
    except Exception as e:
        if config.mapping.rank == 0:
            print(
                f"Skipping {config.strategy.name}: Failed to initialize - {e}")
        return False

    # Iterate over token counts
    num_tokens = config.min_tokens
    while num_tokens <= config.max_tokens:
        try:
            # Skip large fusion ops for very large token counts
            total_elements = num_tokens * config.hidden_size
            if total_elements >= 25600000 and config.fusion_op != AllReduceFusionOp.NONE:
                num_tokens *= config.ratio
                continue

            # Create input tensor with shape (num_tokens, hidden_size)
            input_tensor = torch.ones((num_tokens, config.hidden_size),
                                      dtype=config.torch_dtype,
                                      device="cuda")

            # Capture logs during first allreduce call to detect runtime fallback
            log_lines_run = []
            with capture_trtllm_logs() as log_lines_run:
                # Do a test call to see if it falls back at runtime
                test_params = AllReduceParams(fusion_op=config.fusion_op)

                # First call might trigger runtime fallback detection
                _ = allreduce(input_tensor, all_reduce_params=test_params)

            # Check for runtime fallback
            fallback_msg = check_for_strategy_fallback(log_lines_run,
                                                       config.strategy)
            if fallback_msg:
                if config.mapping.rank == 0:
                    print(
                        f"Skipping {config.strategy.name} (num_tokens={num_tokens}): "
                        f"{fallback_msg}")
                return False

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
                    "all_reduce_params":
                    AllReduceParams(fusion_op=config.fusion_op)
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

        except Exception as e:
            if config.mapping.rank == 0:
                print(
                    f"Skipping {config.strategy.name} (num_tokens={num_tokens}): "
                    f"Runtime error - {e}")
            return False

    return True


@contextlib.contextmanager
def capture_trtllm_logs():
    """
    Capture TensorRT-LLM log output to detect fallback messages.

    Yields:
        A list that will be populated with log lines
    """
    captured_lines = []

    # TRT-LLM logs go to stdout/stderr, so we need to capture both
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class LogCapture:

        def __init__(self, original_stream):
            self.original_stream = original_stream
            self.buffer = io.StringIO()

        def write(self, text):
            self.original_stream.write(text)
            self.original_stream.flush()
            captured_lines.append(text)

        def flush(self):
            self.original_stream.flush()

        def __getattr__(self, name):
            return getattr(self.original_stream, name)

    try:
        # Note: We still write to original streams so user sees output
        sys.stdout = LogCapture(original_stdout)
        sys.stderr = LogCapture(original_stderr)
        yield captured_lines
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def check_for_strategy_fallback(
        log_lines: List[str],
        expected_strategy: AllReduceStrategy) -> Optional[str]:
    """
    Check captured logs for fallback messages indicating strategy couldn't run.

    Args:
        log_lines: List of captured log lines
        expected_strategy: The strategy that was requested

    Returns:
        Error message if fallback detected, None otherwise
    """
    # Look for the telltale fallback messages from C++
    fallback_indicators = [
        "fallback to AllReduceStrategy: NCCL",
        "fallback to AllReduceStrategy:",
        "Since Peer to Peer not supported",
        "Since messageSize is greater than maxWorkspaceSize",
        "Since messageSize > maxWorkspace",
        "Since not aligned",
    ]

    log_text = "".join(log_lines)

    for indicator in fallback_indicators:
        if indicator in log_text:
            # Don't flag NCCL strategy itself as having fallen back
            if expected_strategy == AllReduceStrategy.NCCL:
                continue
            return f"Strategy fell back (detected: '{indicator}')"

    return None


def check_strategy_viable(strategy: AllReduceStrategy,
                          env: EnvironmentSetup) -> Optional[str]:
    """
    Check if a strategy is viable for the current environment.

    Args:
        strategy: The AllReduce strategy to check
        env: Environment setup with system info

    Returns:
        None if viable, or a string describing why it's not viable
    """
    # Check MNNVL specifically
    if strategy == AllReduceStrategy.MNNVL:
        if not MNNVLAllReduce.is_mnnvl(env.mapping, env.torch_dtype):
            return f"MNNVL not supported (requires multi-node, aarch64, supported dtype, and NVLink)"

    # All other strategies should be attempted - we'll catch runtime failures
    return None


def get_default_strategies() -> List[AllReduceStrategy]:
    """
    Get the default list of strategies to test.

    Note: UB strategy is excluded from defaults because it has conflicting
    UB initialization requirements. Use --strategy=UB to test it separately.
    """
    return [
        AllReduceStrategy.NCCL,
        AllReduceStrategy.MIN_LATENCY,
        AllReduceStrategy.NCCL_SYMMETRIC,
        AllReduceStrategy.NCCL_DEVICE,
        AllReduceStrategy.MNNVL,
    ]


def get_default_fusion_ops() -> List[AllReduceFusionOp]:
    """Get the default list of fusion operations to test"""
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
    Skips strategies that are not viable or fail to execute.

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
        # Check if strategy is viable before attempting
        viability_issue = check_strategy_viable(strategy, env)
        if viability_issue:
            if env.rank == 0:
                print(f"Skipping {strategy.name}: {viability_issue}")
            continue

        strategy_failed = False
        for fusion_op in fusion_ops:
            # Skip remaining fusion ops if this strategy already failed
            if strategy_failed:
                break

            config = create_benchmark_config(
                env=env,
                strategy=strategy,
                fusion_op=fusion_op,
                enable_cudagraph=enable_cudagraph,
                inner_loop=inner_loop,
                outer_loop=outer_loop,
                print_header=first_run,
            )
            success = execute_benchmark(config)
            if not success:
                strategy_failed = True
            else:
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
    parser.add_argument(
        "--strategy",
        type=str,
        nargs='*',
        default=None,
        help=
        "Test specific strategies (e.g., --strategy MNNVL NCCL_DEVICE or --strategy MNNVL,NCCL_DEVICE)"
    )
    parser.add_argument(
        "--fusion-op",
        type=str,
        nargs='*',
        default=None,
        help=
        "Test specific fusion ops (e.g., --fusion-op NONE RESIDUAL_RMS_NORM or --fusion-op NONE,RESIDUAL_RMS_NORM)"
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

    # Initialize environment (once, shared across all configs)
    env = initialize_environment(args.dtype, args.token_range, args.hidden_size)

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

        # Parse strategies - support both space-separated and comma-separated
        strategy_names = []
        for arg in args.strategy:
            # Split by comma in case user provides --strategy MNNVL,NCCL
            strategy_names.extend([s.strip() for s in arg.split(',')])

        # Convert to strategy enums
        strategies = []
        for name in strategy_names:
            if name.upper() not in strategy_map:
                raise ValueError(f"Unknown strategy: {name}")
            strategies.append(strategy_map[name.upper()])
    else:
        strategies = get_default_strategies()

    # Determine which fusion ops to test
    if args.fusion_op:
        fusion_map = {
            "NONE": AllReduceFusionOp.NONE,
            "RESIDUAL_RMS_NORM": AllReduceFusionOp.RESIDUAL_RMS_NORM,
        }

        # Parse fusion ops - support both space-separated and comma-separated
        fusion_names = []
        for arg in args.fusion_op:
            # Split by comma in case user provides --fusion-op NONE,RESIDUAL_RMS_NORM
            fusion_names.extend([f.strip() for f in arg.split(',')])

        # Convert to fusion op enums
        fusion_ops = []
        for name in fusion_names:
            if name.upper() not in fusion_map:
                raise ValueError(f"Unknown fusion op: {name}")
            fusion_ops.append(fusion_map[name.upper()])
    else:
        fusion_ops = get_default_fusion_ops()

    # Initialize user buffers based on which strategies are being tested
    initialize_userbuffers_for_strategies(env, strategies)

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
