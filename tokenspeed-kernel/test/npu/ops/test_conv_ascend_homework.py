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

"""Ascend causal convolution output and state tests at Qwen3.5 width four."""

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.conv import causal_conv1d_fn, causal_conv1d_update

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU required"
)


@pytest.mark.parametrize("channels", [32, 6144])
@pytest.mark.parametrize("tokens", [1, 7, 16, 256])
def test_causal_conv_prefill_and_decode_state(tokens, channels):
    torch.manual_seed(31)
    width = 4
    rows = torch.randn(tokens, channels, dtype=torch.bfloat16, device="npu") / 8
    weight = torch.randn(channels, width, dtype=torch.bfloat16, device="npu") / 8
    conv_state = torch.zeros(2, channels, width - 1, dtype=torch.bfloat16, device="npu")
    boundaries = torch.tensor([0, tokens], dtype=torch.int32, device="npu")
    metadata = build_causal_conv1d_prefill_metadata(
        boundaries, torch.tensor([tokens], dtype=torch.int64), 8
    )
    out = causal_conv1d_fn(
        rows.T,
        weight,
        None,
        conv_state,
        boundaries,
        cache_indices=torch.tensor([1], dtype=torch.int32, device="npu"),
        has_initial_state=torch.tensor([False], device="npu"),
        activation="silu",
        prefill_metadata=metadata,
    ).T
    ref = F.silu(
        F.conv1d(rows.T[None], weight[:, None, :], groups=channels, padding=width - 1)[
            0, :, :tokens
        ]
    ).T
    torch.testing.assert_close(out.float(), ref.float(), atol=0.002, rtol=0.02)
    expected_state = F.pad(rows.T, (width - 1, 0))[:, -(width - 1) :]
    torch.testing.assert_close(conv_state[1], expected_state, atol=0, rtol=0)

    next_row = torch.randn(1, channels, dtype=torch.bfloat16, device="npu") / 8
    next_input = next_row.clone()
    before = conv_state[1].clone()
    decode = causal_conv1d_update(
        next_row,
        conv_state,
        weight,
        activation="silu",
        conv_state_indices=torch.tensor([1], dtype=torch.int32, device="npu"),
        output_state_indices=torch.tensor([[1]], dtype=torch.int32, device="npu"),
        parent_indices=None,
    )
    window = torch.cat([before, next_input.T], dim=1)
    ref_decode = F.silu((window.float() * weight.float()).sum(-1)).to(torch.bfloat16)
    torch.testing.assert_close(decode[0], ref_decode, atol=0.002, rtol=0.02)
    torch.testing.assert_close(conv_state[1], window[:, 1:], atol=0, rtol=0)
