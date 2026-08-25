# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.autotuner import autotune


def _make_operands(dtype: torch.dtype):
    torch.manual_seed(0)
    activation = torch.randn((4, 128), dtype=dtype, device="cuda")
    weight = torch.randn((192, 128), dtype=dtype, device="cuda")
    act, act_scale = torch.ops.tensorrt_llm.quantize_e4m3_activation(
        activation)
    weight_scale = weight.abs().amax(dim=1).float() / 448
    weight_fp8 = (weight / weight_scale[:, None]).to(torch.float8_e4m3fn)
    return act, weight_fp8, act_scale.float(), weight_scale


def test_fp8_rowwise_gemm_out_matches_allocating_and_cuda_graph():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 rowwise GEMM requires Hopper or newer")
    operands = _make_operands(torch.bfloat16)
    with torch.inference_mode(), autotune():
        reference = torch.ops.trtllm.fp8_rowwise_gemm(*operands,
                                                       torch.bfloat16)
        base = reference.new_empty((8, reference.size(1)))
        torch.ops.trtllm.fp8_rowwise_gemm_out(
            *operands, base.narrow(0, 0, reference.size(0)))
        eager = base.narrow(0, 0, reference.size(0))
    torch.testing.assert_close(eager, reference, rtol=1e-2, atol=0.15)

    graph_base = reference.new_empty((8, reference.size(1)))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.trtllm.fp8_rowwise_gemm_out(
            *operands, graph_base.narrow(0, 0, reference.size(0)))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_base.narrow(0, 0, reference.size(0)),
                               reference,
                               rtol=1e-2,
                               atol=0.15)
