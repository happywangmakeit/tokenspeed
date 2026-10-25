# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend layernorm kernels."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel_npu.ops.layernorm import qk_rmsnorm as _npu_qk_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import rmsnorm

__all__ = ["gated_rmsnorm", "qk_rmsnorm", "rmsnorm"]


@register_kernel(
    "layernorm",
    "gated_rmsnorm",
    name="ascend_gated_rmsnorm",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
    signatures=format_signatures(
        "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.PORTABLE + 1,
    traits={"fp8_out": frozenset({False})},
)
def gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor | None,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    sigmoid_gate: bool,
    fp8_scale: torch.Tensor | None,
    weights_independent: bool,
) -> torch.Tensor:
    """NPU reference path with FP32 accumulation and grouped statistics."""
    del weights_independent
    if fp8_scale is not None:
        raise ValueError("Ascend gated RMSNorm does not support FP8 output")
    width = x.shape[-1]
    if width % group_size:
        raise ValueError("group_size must divide the final dimension")
    if weight.numel() not in (width, group_size):
        raise ValueError("weight must have width or group_size elements")
    values = x.float().reshape(*x.shape[:-1], width // group_size, group_size)
    if z is not None:
        gate = torch.sigmoid(z.float())
        if not sigmoid_gate:
            gate = gate * z.float()
        gate = gate.reshape_as(values)
        if not norm_before_gate:
            values = values * gate
    variance = values.square().mean(dim=-1, keepdim=True)
    result = values * torch.rsqrt(variance + eps)
    result = result * weight.float().reshape(-1, group_size)
    if z is not None and norm_before_gate:
        result = result * gate
    return result.reshape_as(x).to(x.dtype)


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    *,
    weight_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The NPU per-head norm, with ``weight_offset + weight`` formed in the weight dtype."""
    if weight_offset:
        q_weight = (q_weight.float() + weight_offset).to(q_weight.dtype)
        k_weight = (k_weight.float() + weight_offset).to(k_weight.dtype)
    return _npu_qk_rmsnorm(q, k, q_weight, k_weight, eps)
