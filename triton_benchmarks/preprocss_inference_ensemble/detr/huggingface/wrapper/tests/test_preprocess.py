import numpy as np
import torch
from loguru import logger
from transformers import DetrImageProcessor

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.wrapper.preprocess import (
    detr_preprocessing_fn,
)

FULLHD_WIDTH = 1920
FULLHD_HEIGHT = 1080


def test_detr_preprocessing_fn():
    batch_size = 2
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    batched_image_tensor = torch.randint(
        0, 255, size=(batch_size, 3, FULLHD_HEIGHT, FULLHD_WIDTH), dtype=torch.uint8
    )
    logger.info(f"input tensor={batched_image_tensor.shape}")

    outputs = detr_preprocessing_fn(processor, batched_image_tensor)
    pixcel_values = outputs["pixel_values"]
    pixcel_mask = outputs["pixel_mask"]

    assert isinstance(pixcel_values, torch.Tensor)
    assert pixcel_values.dtype == torch.float32
    np.testing.assert_array_equal(pixcel_values.shape, (batch_size, 3, 750, 1333))
    assert isinstance(pixcel_mask, torch.Tensor)
    assert pixcel_mask.dtype == torch.int64
    np.testing.assert_array_equal(pixcel_mask.shape, (batch_size, 750, 1333))
