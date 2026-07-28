# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.custom_ops.torch_custom_ops import AllReduceRunner
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.modules.embedding import Embedding, LMHead
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.utils import (get_model_extra_attrs,
                                      model_extra_attrs)
from tensorrt_llm.functional import (AllReduceFusionOp, AllReduceStrategy)
from tensorrt_llm.mapping import Mapping


def _allreduce_runner(**overrides):
    options = {
        "tp_size": 2,
        "group": [0, 1],
        "strategy": AllReduceStrategy.AUTO,
        "op": AllReduceFusionOp.NONE,
        "eps": 1e-5,
        "trigger_completion_at_end": True,
        "input_dtype": torch.bfloat16,
        "input_device": torch.device("cuda", 0),
        "input_uses_nccl_window": False,
    }
    options.update(overrides)
    return AllReduceRunner(**options)


@pytest.mark.parametrize(
    "override",
    [
        {
            "group": [2, 3]
        },
        {
            "strategy": AllReduceStrategy.NCCL
        },
        {
            "op": AllReduceFusionOp.RESIDUAL_RMS_NORM
        },
        {
            "eps": 1e-6
        },
        {
            "trigger_completion_at_end": False
        },
        {
            "input_dtype": torch.float16
        },
        {
            "input_device": torch.device("cuda", 1)
        },
        {
            "input_uses_nccl_window": True
        },
    ],
)
def test_allreduce_runner_identity_covers_stable_execution_mode(override):
    assert _allreduce_runner().unique_id() != _allreduce_runner(
        **override).unique_id()


def test_allreduce_runner_identity_leaves_shapes_to_profile_key():
    identity = _allreduce_runner().unique_id()
    assert all("shape" not in field_name for field_name, _ in identity)
    assert all("numel" not in field_name for field_name, _ in identity)


def test_allreduce_inherits_global_strategy_but_explicit_auto_wins():
    mapping = Mapping()
    with model_extra_attrs(
        {"allreduce_strategy": AllReduceStrategy.NCCL_SYMMETRIC}):
        inherited = AllReduce(mapping=mapping)
        explicit = AllReduce(mapping=mapping,
                             strategy=AllReduceStrategy.AUTO)

    assert inherited.strategy == AllReduceStrategy.NCCL_SYMMETRIC
    assert explicit.strategy == AllReduceStrategy.AUTO


@pytest.mark.parametrize(
    "module_factory",
    [
        lambda: Linear(
            8,
            8,
            dtype=torch.float32,
            mapping=Mapping(),
            skip_create_weights_in_init=True,
        ),
        lambda: LMHead(8, 8, dtype=torch.float32, mapping=Mapping()),
        lambda: Embedding(8, 8, dtype=torch.float32, mapping=Mapping()),
    ],
    ids=["linear", "lm_head", "embedding"],
)
def test_nested_allreduce_modules_inherit_global_strategy(
        monkeypatch, module_factory):
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.linear.get_sm_version", lambda: -1)
    with model_extra_attrs({"allreduce_strategy": AllReduceStrategy.NCCL}):
        module = module_factory()

    assert module.all_reduce.strategy == AllReduceStrategy.NCCL


@pytest.mark.parametrize(
    "module_factory",
    [
        lambda: Linear(
            8,
            8,
            dtype=torch.float32,
            mapping=Mapping(),
            allreduce_strategy=AllReduceStrategy.AUTO,
            skip_create_weights_in_init=True,
        ),
        lambda: LMHead(
            8,
            8,
            dtype=torch.float32,
            mapping=Mapping(),
            allreduce_strategy=AllReduceStrategy.AUTO,
        ),
        lambda: Embedding(
            8,
            8,
            dtype=torch.float32,
            mapping=Mapping(),
            allreduce_strategy=AllReduceStrategy.AUTO,
        ),
    ],
    ids=["linear", "lm_head", "embedding"],
)
def test_nested_allreduce_modules_preserve_explicit_override(
        monkeypatch, module_factory):
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.linear.get_sm_version", lambda: -1)
    with model_extra_attrs({"allreduce_strategy": AllReduceStrategy.NCCL}):
        module = module_factory()

    assert module.all_reduce.strategy == AllReduceStrategy.AUTO


def test_glm_mtp_eh_proj_inherits_global_strategy(monkeypatch):
    from tensorrt_llm._torch.models import modeling_glm
    from tensorrt_llm._torch.utils import AuxStreamType

    def _init_decoder_base(module, *_args, **_kwargs):
        nn.Module.__init__(module)

    monkeypatch.setattr(modeling_glm.Glm4DecoderLayer, "__init__",
                        _init_decoder_base)
    monkeypatch.setattr(modeling_glm, "RMSNorm",
                        lambda **_kwargs: nn.Identity())
    monkeypatch.setattr(modeling_glm, "DeepseekV3MTPHead",
                        lambda _config: nn.Identity())
    monkeypatch.setattr(modeling_glm.torch.cuda, "Event", lambda: object())
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.linear.get_sm_version", lambda: -1)

    pretrained_config = SimpleNamespace(
        hidden_size=8,
        moe_intermediate_size=16,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        rms_norm_eps=1e-5,
        torch_dtype=torch.float32,
    )
    model_config = SimpleNamespace(
        pretrained_config=pretrained_config,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    aux_stream = object()

    with model_extra_attrs({"allreduce_strategy": AllReduceStrategy.NCCL}):
        mtp = modeling_glm.Glm4MTP(
            model_config,
            layer_idx=0,
            aux_stream_dict={AuxStreamType.MoeShared: aux_stream},
        )

    assert mtp.eh_proj.all_reduce.strategy == AllReduceStrategy.NCCL


def test_auto_model_publishes_strategy_in_construction_scope(monkeypatch):

    class ScopedStrategyModel(nn.Module):

        def __init__(self, _config):
            super().__init__()
            self.scoped_strategy = get_model_extra_attrs()[
                "allreduce_strategy"]

    config = ModelConfig(
        pretrained_config=SimpleNamespace(
            architectures=["ScopedStrategyModel"],
            is_encoder_decoder=False,
        ),
        allreduce_strategy=AllReduceStrategy.NCCL_SYMMETRIC,
    )
    monkeypatch.setattr(AutoModelForCausalLM, "_resolve_class",
                        lambda _config: ScopedStrategyModel)

    model = AutoModelForCausalLM.from_config(config)

    assert model.scoped_strategy == AllReduceStrategy.NCCL_SYMMETRIC
    assert config.extra_attrs[
        "allreduce_strategy"] == AllReduceStrategy.NCCL_SYMMETRIC
