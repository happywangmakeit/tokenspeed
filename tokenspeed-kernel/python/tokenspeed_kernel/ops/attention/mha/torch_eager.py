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

"""Portable noncausal attention preserving eager rounding boundaries."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attention",
    "vision_eager",
    name="torch_vision_eager",
    solution="torch",
    capability=CapabilityRequirement(),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.PORTABLE,
)
def torch_vision_eager(q, k, v, *, cu_seqlens, scale):
    """Apply full attention independently to each packed sequence on its device.

    Q/K/V are [tokens, heads, dim]; cu_seqlens partitions tokens. Scale is
    applied after QK is rounded to the input dtype. Softmax accumulates in
    FP32 and rounds before multiplying V, matching ordinary eager attention.
    This correctness baseline materializes each sequence's score matrix and
    reads boundaries on the host. It is not a graph-capture implementation.
    """
    bounds = cu_seqlens.cpu().tolist()
    if not bounds or bounds[0] != 0 or bounds[-1] != q.shape[0]:
        raise ValueError("Vision boundaries must cover all tokens")
    if any(b < a for a, b in zip(bounds[:-1], bounds[1:])):
        raise ValueError("Vision boundaries must be nondecreasing")
    outputs = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if start == end:
            continue
        query, key, value = [
            x[start:end].transpose(0, 1).unsqueeze(0) for x in (q, k, v)
        ]
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        outputs.append(torch.matmul(probabilities, value).squeeze(0).transpose(0, 1))
    return torch.cat(outputs) if outputs else torch.empty_like(q)
