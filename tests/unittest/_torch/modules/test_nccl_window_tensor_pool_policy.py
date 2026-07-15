import pytest

from tensorrt_llm._torch.modules.linear import \
    is_nccl_window_tensor_pool_method_enabled


@pytest.mark.parametrize(
    "method, expected",
    [
        ("nvfp4", True),
        ("fp8_rowwise", False),
        ("cublas_mm", False),
        (None, False),
    ],
)
def test_default_pool_method_policy(monkeypatch, method, expected):
    monkeypatch.delenv("TLLM_NCCL_WINDOW_TENSOR_POOL_METHODS", raising=False)
    assert is_nccl_window_tensor_pool_method_enabled(method) is expected


@pytest.mark.parametrize(
    "setting, method, expected",
    [
        ("fp8_rowwise", "nvfp4", False),
        ("fp8_rowwise", "fp8_rowwise", True),
        ("cublas_mm", "cublas_mm", True),
        ("nvfp4, fp8_rowwise", "nvfp4", True),
        ("nvfp4, fp8_rowwise", "fp8_rowwise", True),
        ("all", "fp8_rowwise", True),
        ("all", "cublas_mm", True),
        ("none", "nvfp4", False),
    ],
)
def test_pool_method_policy_override(monkeypatch, setting, method, expected):
    monkeypatch.setenv("TLLM_NCCL_WINDOW_TENSOR_POOL_METHODS", setting)
    assert is_nccl_window_tensor_pool_method_enabled(method) is expected
