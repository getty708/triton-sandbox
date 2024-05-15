import torch
from loguru import logger
from transformers import DetrForObjectDetection, DetrImageProcessor

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    DETR_LOGITS_TENSOR_NAME,
    DETR_PIXEL_VALUES_TENSOR_NAME,
    DETR_PRED_BOXES_TENSOR_NAME,
    HUGGINGFACE_MODEL_NAME,
    HUGGINGFACE_MODEL_REVISION,
)
from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.wrapper.postprocess import (
    detr_postprocessing_fn,
)
from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.wrapper.preprocess import (
    detr_preprocessing_fn,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_detr_model() -> tuple[DetrForObjectDetection, DetrImageProcessor]:
    processor = DetrImageProcessor.from_pretrained(
        HUGGINGFACE_MODEL_NAME, revision=HUGGINGFACE_MODEL_REVISION
    )
    model = DetrForObjectDetection.from_pretrained(
        HUGGINGFACE_MODEL_NAME, revision=HUGGINGFACE_MODEL_REVISION
    ).to(device=_DEVICE)
    logger.info("DETR model and processor are initialized.")
    return model, processor


def detr_end2end_inference(
    model: DetrForObjectDetection,
    processor: DetrImageProcessor,
    batched_image: torch.Tensor,
) -> dict[str, torch.Tensor]:
    preproc_outputs: dict[str, torch.Tensor] = detr_preprocessing_fn(
        processor,
        batched_image,
    )
    outputs: dict[str, torch.Tensor] = model(
        pixel_values=preproc_outputs[DETR_PIXEL_VALUES_TENSOR_NAME].to(device=model.device),
        return_dict=True,
    )
    postproc_outputs: dict[str, torch.Tensor] = detr_postprocessing_fn(
        processor,
        logits_batch=outputs[DETR_LOGITS_TENSOR_NAME],
        pred_boxes_batch=outputs[DETR_PRED_BOXES_TENSOR_NAME],
    )
    return postproc_outputs
