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

"""Real Qwen vision modules against independently constructed Transformers."""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.qwen3_vision import Qwen3VLMoeVisionModel


@pytest.mark.parametrize(
    "grid", [[[1, 2, 2]], [[1, 4, 6]], [[2, 4, 4]], [[1, 2, 2], [1, 4, 6]]]
)
@torch.inference_mode()
def test_vision_bf16_eager_matches_reference(grid):
    torch.manual_seed(2026)
    config = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=64,
        spatial_merge_size=2,
        deepstack_visual_indexes=[],
    )
    config._attn_implementation = "eager"
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        reference = Qwen3_5VisionModel(config).eval().to("npu")
        with torch.device("npu"):
            actual = Qwen3VLMoeVisionModel(
                config,
                Mapping(rank=0, world_size=1),
                mm_attention_backend="torch_eager",
            ).eval()
    finally:
        torch.set_default_dtype(previous)
    actual.load_state_dict(
        {
            k.replace("attn.qkv.", "attn.qkv_proj."): v
            for k, v in reference.state_dict().items()
        },
        strict=True,
    )
    positions = torch.tensor(grid)
    pixels = torch.randn(
        sum(t * h * w for t, h, w in grid), 24, device="npu", dtype=torch.bfloat16
    )
    result = actual.forward_blocks(
        actual.prepare_patch_embed(pixels, positions),
        actual.prepare_metadata(positions),
    )
    expected = reference(pixels, positions.to("npu")).pooler_output
    torch.testing.assert_close(result, expected, atol=0.002, rtol=0.02)
    diff = (result.float() - expected.float()).norm()
    assert diff <= 0.02 * expected.float().norm()
