"""Qwen3.5 head_dim=256 Ascend attention against independent CPU attention."""

import itertools
import math

import torch
from tokenspeed_kernel_npu.ops.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)


def reference(q, k, v, offset=0, causal=True):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) / math.sqrt(256)
    if causal:
        positions = torch.arange(k.shape[0])
        queries = offset + torch.arange(q.shape[0])
        scores = scores.masked_fill(
            positions[None, None] > queries[None, :, None], -torch.inf
        )
    probs = scores.softmax(-1)
    return torch.einsum("hts,shd->thd", probs, v.float()).to(q.dtype)


def test_prefill_ragged_256():
    torch.manual_seed(11)
    q = torch.randn(7, 8, 256, dtype=torch.bfloat16)
    k = torch.randn(7, 2, 256, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    boundaries = [0, 3, 7]
    expected = torch.cat(
        [reference(q[a:b], k[a:b], v[a:b]) for a, b in itertools.pairwise(boundaries)]
    )
    actual = mha_prefill(
        q.npu(),
        k.npu(),
        v.npu(),
        torch.tensor(boundaries, device="npu"),
        boundaries,
        4,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)


def test_paged_extend_decode_256():
    torch.manual_seed(12)
    k_cache = torch.randn(4, 4, 2, 256, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    page_table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
    lengths = torch.tensor([5, 3], dtype=torch.int32)
    keys = [
        torch.cat([k_cache[p] for p in page_table[i]])[: lengths[i]] for i in range(2)
    ]
    values = [
        torch.cat([v_cache[p] for p in page_table[i]])[: lengths[i]] for i in range(2)
    ]

    q = torch.randn(3, 8, 256, dtype=torch.bfloat16)
    expected = torch.cat(
        [
            reference(q[:2], keys[0], values[0], offset=3),
            reference(q[2:], keys[1], values[1], offset=2),
        ]
    )
    actual = mha_extend_with_kvcache(
        q.npu(),
        torch.tensor([0, 2, 3], device="npu"),
        torch.tensor([0, 5, 8], device="npu"),
        k_cache.npu(),
        v_cache.npu(),
        page_table.npu(),
        lengths.npu(),
        2,
        5,
        is_causal=True,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)

    q_decode = torch.randn(2, 8, 256, dtype=torch.bfloat16)
    decode_expected = torch.cat(
        [
            reference(q_decode[i : i + 1], keys[i], values[i], causal=False)
            for i in range(2)
        ]
    )
    decode = mha_decode_with_kvcache(
        q_decode.npu(),
        k_cache.npu(),
        v_cache.npu(),
        page_table.npu(),
        lengths.npu(),
        5,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(decode.cpu(), decode_expected, atol=2e-2, rtol=2e-2)
