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

"""Correctness-first Ascend GDN scan using NPU PyTorch operators.

The cache is K-last [page, head, value, key]. The recurrence computes in
float32, as Qwen3.5's reference implementation does. Reading the variable
length boundaries on the host synchronizes the NPU once per prefill batch.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.gdn import (
    GdnChunkPrefillResult,
    GdnStateCheckpoints,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def _scan_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    scale: float,
    qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One delta update; inputs may have leading batch dimensions."""
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    state = state.float()
    if q.shape[-2] != v.shape[-2]:
        if v.shape[-2] % q.shape[-2]:
            raise ValueError("GDN value heads must be divisible by query heads")
        repeats = v.shape[-2] // q.shape[-2]
        q = q.repeat_interleave(repeats, dim=-2)
        k = k.repeat_interleave(repeats, dim=-2)
    if qk_l2norm:
        q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    q = q * scale
    state = state * g.exp().unsqueeze(-1).unsqueeze(-1)
    predicted = (state * k.unsqueeze(-2)).sum(dim=-1)
    delta = (v - predicted) * beta.unsqueeze(-1)
    state = state + delta.unsqueeze(-1) * k.unsqueeze(-2)
    out = (state * q.unsqueeze(-2)).sum(dim=-1)
    return out, state


@register_kernel(
    "attention",
    "gdn_chunk_prefill",
    name="ascend_gdn_chunk_prefill",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "output_h": frozenset({False}),
        "qk_l2norm": frozenset({False, True}),
        "state_checkpoints": frozenset({False}),
    },
)
def ascend_gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool = False,
    output_final_state: bool = True,
    output_h: bool = False,
    state_checkpoints: GdnStateCheckpoints | None = None,
) -> GdnChunkPrefillResult:
    """Scan packed request segments on NPU and return the final K-last states."""
    if output_h or state_checkpoints is not None:
        raise ValueError("Ascend GDN does not produce intermediate checkpoints")
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Ascend GDN prefill expects [1, total_tokens, heads, dim]")
    boundaries = cu_seqlens.to("cpu").tolist()
    if boundaries[0] != 0 or boundaries[-1] != q.shape[1]:
        raise ValueError("GDN sequence boundaries do not cover the input")
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    outputs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for row, (start, end) in enumerate(itertools.pairwise(boundaries)):
        state = initial_state[row]
        for token in range(start, end):
            output, state = _scan_step(
                q[0, token],
                k[0, token],
                v[0, token],
                g[0, token],
                beta[0, token],
                state,
                scale=scale,
                qk_l2norm=qk_l2norm,
            )
            outputs.append(output.to(q.dtype))
        states.append(state.to(initial_state.dtype))
    out = (
        torch.stack(outputs, dim=0).unsqueeze(0)
        if outputs
        else q.new_empty((1, 0, v.shape[-2], v.shape[-1]))
    )
    final_state = torch.stack(states, dim=0) if output_final_state else None
    return GdnChunkPrefillResult(out=out, final_state=final_state)


@register_kernel(
    "attention",
    "gdn_decode_step",
    name="ascend_gdn_decode_step",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16}
    ),
    priority=Priority.PORTABLE,
    traits={"strided_qkv": frozenset({False, True})},
)
def ascend_gdn_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    output_state_indices: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Decode one token per request, reading and writing scheduler owned pages."""
    if q.ndim != 4 or q.shape[1] != 1:
        raise ValueError("Ascend GDN decode expects [batch, 1, heads, dim]")
    read_rows = initial_state_indices.to("cpu").tolist()
    write_rows = (
        output_state_indices.to("cpu").tolist()
        if output_state_indices is not None
        else read_rows
    )
    safe_rows = torch.tensor(
        [max(0, row) for row in read_rows], device=q.device, dtype=torch.long
    )
    state = initial_state.index_select(0, safe_rows)
    gate = -A_log.float().exp() * F.softplus(a[:, 0].float() + dt_bias.float())
    beta = b[:, 0].float().sigmoid()
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    output, next_state = _scan_step(
        q[:, 0],
        k[:, 0],
        v[:, 0],
        gate,
        beta,
        state,
        scale=scale,
        qk_l2norm=use_qk_l2norm,
    )
    for row, destination in enumerate(write_rows):
        if destination >= 0 and read_rows[row] >= 0:
            initial_state[destination].copy_(next_state[row].to(initial_state.dtype))
        else:
            output[row].zero_()
    return output.unsqueeze(1).to(q.dtype)
