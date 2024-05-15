import numpy as np
import pytest
import torch

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    NUM_MAX_DETECTION,
)
from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.wrapper.detection_service import (
    detr_end2end_inference,
    init_detr_model,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("batch_size", [1, 2])
def test_detr_inference_with_helpers(batch_size: int):
    image_batch = torch.randint(
        0, 255, (batch_size, 3, 1080, 1920), dtype=torch.uint8, device=_DEVICE
    )

    model, processor = init_detr_model()
    postproc_outputs = detr_end2end_inference(model, processor, image_batch)

    scores = postproc_outputs["scores"]
    labels = postproc_outputs["labels"]
    boxes = postproc_outputs["boxes"]
    assert isinstance(scores, torch.Tensor)
    assert scores.dtype == torch.float32
    np.testing.assert_array_equal(scores.size(), (batch_size, NUM_MAX_DETECTION))
    assert isinstance(labels, torch.Tensor)
    assert labels.dtype == torch.int64
    np.testing.assert_array_equal(labels.size(), (batch_size, NUM_MAX_DETECTION))
    assert isinstance(boxes, torch.Tensor)
    assert boxes.dtype == torch.float32
    np.testing.assert_array_equal(boxes.size(), (batch_size, NUM_MAX_DETECTION, 4))
