import torch
from transformers import DetrImageProcessor
from transformers.image_processing_utils import BatchFeature

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    DETR_PIXEL_VALUES_TENSOR_NAME,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detr_preprocessing_fn(
    processor: DetrImageProcessor,
    image_batch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply preprocessing for the DETR model.

    Args:
        processor (DetrImageProcessor): instance of DetrImageProcessor().
        image_batch (torch.Tensor): batched image tensor. shape = (batch_size, 3, H, W)

    Returns:
        tensors of pixel_values and pixel_mask.
    """
    assert (
        image_batch.size(1) == 3
    ), f"Expected channel first format, but got {image_batch.size()}"
    # TODO: Do preprocessing with torch. DetrImageProcessor does preprocessing with numpy and unncesary conversion is performced..
    preproc_outputs: BatchFeature = processor(
        images=image_batch,
        return_tensors="pt",
        data_format="channels_first",
        input_data_format="channels_first",
    )
    # Output from DetrImageProcessor is in CPU memory. Move them to GPU.
    input_tensors: dict[str, torch.Tensor] = preproc_outputs.data
    input_tensors[DETR_PIXEL_VALUES_TENSOR_NAME] = input_tensors[
        DETR_PIXEL_VALUES_TENSOR_NAME
    ].to(device=_DEVICE)
    return input_tensors
