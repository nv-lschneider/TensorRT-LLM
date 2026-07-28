# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch import cuda_graph_utils


class _FakeCuda:

    def __init__(self, calls):
        self.calls = calls
        self.stream = 17

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def current_stream(self, _device):
        return SimpleNamespace(cuda_stream=self.stream)

    @contextlib.contextmanager
    def graph(self, _graph, *, pool=None):
        self.calls.append(("graph_enter", pool))
        try:
            yield
        finally:
            self.calls.append(("graph_exit", pool))


def _install_fake_runtime(monkeypatch):
    calls = []
    next_handle = iter((101, 102, 103))

    def record(name, result=None):

        def op(*args):
            calls.append((name, *args))
            return result() if callable(result) else result

        return op

    fake_cuda = _FakeCuda(calls)
    fake_ops = SimpleNamespace(
        trtllm=SimpleNamespace(
            _create_nccl_window_reuse_domain=record("create",
                                                    lambda: next(next_handle)),
            _begin_nccl_window_preparation=record("begin_prepare"),
            _end_nccl_window_preparation=record("end_prepare"),
            _begin_nccl_window_capture=record("begin_capture", 501),
            _end_nccl_window_capture=record("end_capture"),
            _quiesce_nccl_window_reuse_domain=record("quiesce"),
            _close_nccl_window_reuse_domain=record("close"),
        ))
    fake_torch = SimpleNamespace(cuda=fake_cuda,
                                 ops=fake_ops,
                                 device=cuda_graph_utils.torch.device)
    monkeypatch.setattr(cuda_graph_utils, "torch", fake_torch)
    return calls, fake_cuda


def test_domain_lifecycle_is_prepare_capture_quiesce_close(monkeypatch):
    calls, _ = _install_fake_runtime(monkeypatch)
    domain = cuda_graph_utils.NCCLWindowReuseDomain()

    with domain.prepare() as handle:
        assert handle == 101
        assert cuda_graph_utils.get_active_nccl_window_reuse_domain_id() == 101

    with domain.capture(object(), pool="pool"):
        assert cuda_graph_utils.get_active_nccl_window_reuse_domain_id() == 101

    domain.quiesce()
    domain.close()
    domain.close()

    assert calls == [
        ("create", 0),
        ("begin_prepare", 101),
        ("end_prepare", 101),
        ("begin_prepare", 101),
        ("graph_enter", "pool"),
        ("begin_capture", 101),
        ("end_capture", 501),
        ("graph_exit", "pool"),
        ("end_prepare", 101),
        ("quiesce", 101),
        ("close", 101),
    ]


def test_nested_borrow_is_explicit_and_lane_checked(monkeypatch):
    _, fake_cuda = _install_fake_runtime(monkeypatch)
    owner = cuda_graph_utils.NCCLWindowReuseDomain()
    borrower = cuda_graph_utils.NCCLWindowReuseDomain(borrow_active=True)
    foreign_owner = cuda_graph_utils.NCCLWindowReuseDomain()

    with owner.prepare() as owner_handle:
        with borrower.prepare() as borrowed_handle:
            assert borrowed_handle == owner_handle
        with pytest.raises(RuntimeError, match="different NCCL window"):
            with foreign_owner.prepare():
                pass

        fake_cuda.stream = 23
        with pytest.raises(RuntimeError, match="different CUDA stream"):
            with borrower.prepare():
                pass

    owner.quiesce()
    owner.close()


def test_close_rotates_to_a_fresh_native_generation(monkeypatch):
    calls, _ = _install_fake_runtime(monkeypatch)
    domain = cuda_graph_utils.NCCLWindowReuseDomain()

    with domain.prepare() as first:
        assert first == 101
    domain.quiesce()
    with pytest.raises(RuntimeError, match="quiesced"):
        with domain.prepare():
            pass
    domain.close()

    with domain.prepare() as second:
        assert second == 102
    domain.quiesce()
    domain.close()

    assert [call for call in calls if call[0] == "create"] == [
        ("create", 0),
        ("create", 0),
    ]
