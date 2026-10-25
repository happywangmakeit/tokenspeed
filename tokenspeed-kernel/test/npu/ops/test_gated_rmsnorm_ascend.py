"""Ascend grouped gated RMSNorm checks against a CPU FP32 reference."""

import pytest
import torch
from tokenspeed_kernel.ops.layernorm import gated_rmsnorm


@pytest.mark.parametrize(
    "shape,group_size,shared_weight,norm_before_gate,sigmoid_gate,use_gate",
    [
        ((1, 2048), 128, True, True, False, True),
        ((3, 256), 64, False, False, True, True),
        ((0, 128), 128, True, True, False, False),
        ((2, 128), 128, True, True, False, False),
    ],
)
def test_gated_rmsnorm_ascend(
    shape, group_size, shared_weight, norm_before_gate, sigmoid_gate, use_gate
):
    torch.manual_seed(17)
    x = torch.randn(shape, dtype=torch.bfloat16)
    weight = torch.randn(
        group_size if shared_weight else shape[-1], dtype=torch.bfloat16
    )
    z = torch.randn(shape, dtype=torch.bfloat16) if use_gate else None
    eps = 1e-6

    values = x.float().reshape(*shape[:-1], shape[-1] // group_size, group_size)
    if z is not None:
        gate = torch.sigmoid(z.float())
        if not sigmoid_gate:
            gate = gate * z.float()
        gate = gate.reshape_as(values)
        if not norm_before_gate:
            values = values * gate
    reference = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    reference = reference * weight.float().reshape(-1, group_size)
    if z is not None and norm_before_gate:
        reference = reference * gate
    reference = reference.reshape(shape).to(x.dtype)

    actual = gated_rmsnorm(
        x.npu(),
        weight.npu(),
        None if z is None else z.npu(),
        eps=eps,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        fp8_scale=None,
        weights_independent=True,
    )
    torch.npu.synchronize()
    assert actual.shape == reference.shape
    if actual.numel():
        torch.testing.assert_close(actual.cpu(), reference, atol=2e-2, rtol=2e-2)
