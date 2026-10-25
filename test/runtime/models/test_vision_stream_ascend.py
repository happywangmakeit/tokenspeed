"""Multimodal feature release preserves asynchronous NPU consumers."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.multimodal.embedder import _drop_prefilled_encodings
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem


@pytest.mark.skipif(not current_platform().is_npu, reason="Ascend NPU required")
@pytest.mark.parametrize("deepstack", [False, True])
def test_drop_prefilled_encodings_on_npu(deepstack):
    producer = torch.npu.Stream()
    consumer = torch.npu.Stream()
    for value in range(1, 9):
        with torch.npu.stream(producer):
            encoded = torch.full((1024, 1024), float(value), device="npu")
            extra = encoded + 1 if deepstack else None
        consumer.wait_stream(producer)
        item = MultimodalDataItem(
            modality=Modality.IMAGE, encoded=encoded, encoded_deepstack=extra
        )
        del encoded, extra
        with torch.npu.stream(consumer):
            result = item.encoded * 2
            if deepstack:
                result += item.encoded_deepstack
            _drop_prefilled_encodings([item])
        assert item.encoded is None and item.encoded_deepstack is None
        with torch.npu.stream(producer):
            churn = [torch.full((1024, 1024), -99.0, device="npu") for _ in range(4)]
        consumer.synchronize()
        expected = float(value * 3 + 1 if deepstack else value * 2)
        torch.testing.assert_close(
            result.cpu(), torch.full((1024, 1024), expected), rtol=0, atol=0
        )
        del churn
