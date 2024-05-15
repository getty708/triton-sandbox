import numpy as np
import pytest
import torch
from transformers import DetrImageProcessor

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.wrapper.postprocess import (
    NUM_MAX_DETECTION,
    detr_postprocessing_fn,
)


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def logits_batch(batch_size: int) -> torch.Tensor:
    return torch.normal(0, 1, size=(batch_size, 100, 92)) * 10.0


@pytest.fixture
def pred_boxes_batch(batch_size: int) -> torch.Tensor:
    return torch.rand(batch_size, 100, 4)


def test_detr_postprocessing_fn(
    batch_size: int,
    logits_batch: torch.Tensor,
    pred_boxes_batch: torch.Tensor,
):
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    outputs_batch: dict[str, torch.Tensor] = detr_postprocessing_fn(
        processor, logits_batch, pred_boxes_batch
    )

    assert set(outputs_batch.keys()) == set(["scores", "labels", "boxes"])
    np.testing.assert_array_equal(
        outputs_batch["scores"].size(), (batch_size, NUM_MAX_DETECTION)
    )
    np.testing.assert_array_equal(
        outputs_batch["labels"].size(), (batch_size, NUM_MAX_DETECTION)
    )
    np.testing.assert_array_equal(
        outputs_batch["boxes"].size(), (batch_size, NUM_MAX_DETECTION, 4)
    )
