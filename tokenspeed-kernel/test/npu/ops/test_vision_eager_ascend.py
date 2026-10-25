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

"""Packed noncausal eager attention, including empty and reordered requests."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha import vision_eager


@pytest.mark.parametrize("lengths", [[1], [3, 0, 7], [16, 256]])
@torch.inference_mode()
def test_eager_vision_attention_request_isolation(lengths):
    torch.manual_seed(31)
    count = sum(lengths)
    q, k, v = [
        torch.randn(count, 12, 64, dtype=torch.bfloat16, device="npu") for _ in range(3)
    ]
    bounds = [0]
    for n in lengths:
        bounds.append(bounds[-1] + n)
    actual = vision_eager(
        q, k, v, cu_seqlens=torch.tensor(bounds, device="npu"), scale=0.125
    )
    references = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if start == end:
            continue
        # Per-head CPU FP32 math is independent of the packed NPU implementation.
        qq, kk, vv = [x[start:end].cpu().float().transpose(0, 1) for x in (q, k, v)]
        scores = (qq @ kk.transpose(-1, -2)).to(torch.bfloat16)
        scores = (scores * 0.125).float()
        probs = scores.softmax(-1).to(torch.bfloat16).float()
        references.append((probs @ vv).transpose(0, 1).to(torch.bfloat16))
    torch.testing.assert_close(
        actual.cpu(), torch.cat(references), atol=0.002, rtol=0.02
    )
    parts = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        parts.append(
            vision_eager(
                q[start:end],
                k[start:end],
                v[start:end],
                cu_seqlens=torch.tensor([0, end - start], device="npu"),
                scale=0.125,
            )
        )
    torch.testing.assert_close(actual, torch.cat(parts), atol=0, rtol=0)
