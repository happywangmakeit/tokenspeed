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

"""Ascend GDN scan against the Transformers float32 reference recurrence."""

import itertools

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU required"
)


def _reference(q, k, v, g, beta, initial):
    out, state = torch_recurrent_gated_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        g.float(),
        beta.float(),
        initial_state=initial.float().transpose(-1, -2),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    return out, state.transpose(-1, -2)


@pytest.mark.parametrize("heads,width", [(2, 8), (16, 128)])
@pytest.mark.parametrize("lengths", [(1,), (3, 0, 4), (16,), (256,)])
def test_prefill_matches_reference_and_keeps_requests_isolated(lengths, heads, width):
    torch.manual_seed(13)
    tokens = sum(lengths)
    q = torch.randn(1, tokens, heads, width, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = -torch.rand(1, tokens, heads, dtype=torch.float32)
    beta = torch.rand(1, tokens, heads, dtype=torch.float32)
    initial = torch.randn(len(lengths), heads, width, width, dtype=torch.float32) / 10
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    result = gdn_chunk_prefill(
        q.npu(),
        k.npu(),
        v.npu(),
        g.npu(),
        beta.npu(),
        scale=width**-0.5,
        initial_state=initial.npu(),
        cu_seqlens=torch.tensor(bounds, dtype=torch.int32, device="npu"),
        qk_l2norm=True,
    )
    expected_out = []
    expected_states = []
    for row, (start, end) in enumerate(itertools.pairwise(bounds)):
        if start == end:
            expected_states.append(initial[row])
            continue
        out, state = _reference(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            initial[row : row + 1],
        )
        expected_out.append(out)
        expected_states.append(state[0])
    torch.testing.assert_close(
        result.out.cpu().float(), torch.cat(expected_out, dim=1), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        result.final_state.cpu(), torch.stack(expected_states), rtol=0.02, atol=0.02
    )


@pytest.mark.parametrize("heads,width", [(2, 8), (16, 128)])
def test_decode_writes_destination_without_changing_source_or_padding(heads, width):
    torch.manual_seed(17)
    q = torch.randn(2, 1, heads, width, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(2, 1, heads, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(heads, dtype=torch.float32)
    dt_bias = torch.randn(heads, dtype=torch.float32)
    pool = torch.randn(3, heads, width, width, dtype=torch.float32) / 10
    before = pool.clone()
    device_pool = pool.npu()
    output = gdn_decode_step(
        q.npu(),
        k.npu(),
        v.npu(),
        A_log=a_log.npu(),
        a=a.npu(),
        dt_bias=dt_bias.npu(),
        b=b.npu(),
        initial_state=device_pool,
        initial_state_indices=torch.tensor([0, -1], dtype=torch.int32, device="npu"),
        output_state_indices=torch.tensor([2, -1], dtype=torch.int32, device="npu"),
        scale=width**-0.5,
        use_qk_l2norm=True,
    )
    gate = -a_log.exp() * torch.nn.functional.softplus(a[0, 0].float() + dt_bias)
    beta = b[0:1].float().sigmoid()
    expected_out, expected_state = _reference(
        q[0:1], k[0:1], v[0:1], gate[None, None], beta, before[0:1]
    )
    torch.testing.assert_close(
        output[0:1].cpu().float(), expected_out, rtol=0.02, atol=0.02
    )
    assert torch.count_nonzero(output[1]).item() == 0
    updated = device_pool.cpu()
    torch.testing.assert_close(updated[0], before[0])
    torch.testing.assert_close(updated[1], before[1])
    torch.testing.assert_close(updated[2], expected_state[0], rtol=0.02, atol=0.02)


@pytest.mark.parametrize("tokens", [16, 256])
def test_model_geometry_split_prefill_and_eight_decode_steps(tokens):
    """Carry real-size state across chunks and eight scheduler-style updates."""
    torch.manual_seed(20261026)
    heads, width, steps = 16, 128, 8
    q = torch.randn(1, tokens + steps, heads, width, dtype=torch.bfloat16) / 4
    k, v = torch.randn_like(q) / 4, torch.randn_like(q) / 4
    a = torch.randn(1, tokens + steps, heads, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(heads, dtype=torch.float32) / 4
    dt_bias = torch.randn(heads, dtype=torch.float32) / 4
    g = -a_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    beta = b.float().sigmoid()
    initial = torch.randn(1, heads, width, width, dtype=torch.float32) / 10
    expected, expected_state = _reference(q, k, v, g, beta, initial)
    state = initial.npu()
    outputs = []
    for start, end in [(0, tokens // 2), (tokens // 2, tokens)]:
        result = gdn_chunk_prefill(
            q[:, start:end].npu(),
            k[:, start:end].npu(),
            v[:, start:end].npu(),
            g[:, start:end].npu(),
            beta[:, start:end].npu(),
            scale=width**-0.5,
            initial_state=state,
            cu_seqlens=torch.tensor([0, end - start], dtype=torch.int32, device="npu"),
            qk_l2norm=True,
        )
        outputs.append(result.out.cpu())
        state = result.final_state
    sentinel = torch.randn_like(state)
    pool = torch.cat([sentinel, state, sentinel.clone()])
    protected = pool[[0, 2]].clone()
    for pos in range(tokens, tokens + steps):
        out = gdn_decode_step(
            q[:, pos : pos + 1].npu(),
            k[:, pos : pos + 1].npu(),
            v[:, pos : pos + 1].npu(),
            A_log=a_log.npu(),
            a=a[:, pos : pos + 1].npu(),
            dt_bias=dt_bias.npu(),
            b=b[:, pos : pos + 1].npu(),
            initial_state=pool,
            initial_state_indices=torch.tensor([1], dtype=torch.int32, device="npu"),
            output_state_indices=torch.tensor([1], dtype=torch.int32, device="npu"),
            scale=width**-0.5,
            use_qk_l2norm=True,
        )
        outputs.append(out.cpu())
    torch.testing.assert_close(
        torch.cat(outputs, dim=1).float(), expected, atol=0.002, rtol=0.02
    )
    torch.testing.assert_close(pool[1:2].cpu(), expected_state, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(pool[[0, 2]], protected, atol=0, rtol=0)
