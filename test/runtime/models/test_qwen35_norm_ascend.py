"""Qwen3.5 Gemma normalization on Ascend matches its FP32 definition."""

import torch

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm


def test_gemma_rmsnorm_ascend_with_and_without_residual():
    torch.manual_seed(2026)
    x = torch.randn(3, 1024, dtype=torch.bfloat16, device="npu")
    residual = torch.randn_like(x)
    norm = GemmaRMSNorm(1024, eps=1e-6).to(device="npu", dtype=torch.bfloat16)
    norm.weight.data.copy_(torch.randn_like(norm.weight) / 8)

    plain = norm(x)
    source = x.float()
    expected = (
        source
        * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-6)
        * (1 + norm.weight.float())
    ).to(x.dtype)
    torch.testing.assert_close(plain.cpu(), expected.cpu(), atol=2e-2, rtol=2e-2)

    added = x + residual
    actual, next_residual = norm(x, residual)
    source = added.float()
    expected = (
        source
        * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-6)
        * (1 + norm.weight.float())
    ).to(x.dtype)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(next_residual.cpu(), added.cpu(), atol=0, rtol=0)
