import torch
from transformers import DetrImageProcessor
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    NUM_MAX_DETECTION,
)


def detr_postprocessing_fn(
    processor: DetrImageProcessor,
    logits_batch: torch.Tensor,
    pred_boxes_batch: torch.Tensor,
    frame_size: tuple[int, int] = (1080, 1920),
):
    batch_size: int = logits_batch.size(0)
    detr_outputs = DetrObjectDetectionOutput(
        logits=logits_batch, pred_boxes=pred_boxes_batch
    )
    input_frame_shape = [frame_size] * batch_size
    outputs = processor.post_process_object_detection(
        detr_outputs, target_sizes=input_frame_shape, threshold=0.9
    )

    device = logits_batch.device
    scores = torch.full(
        (batch_size, NUM_MAX_DETECTION), -1.0, dtype=torch.float32, device=device
    )
    labels = torch.full(
        (batch_size, NUM_MAX_DETECTION), -1, dtype=torch.int64, device=device
    )
    boxes = torch.full(
        (batch_size, NUM_MAX_DETECTION, 4), -1.0, dtype=torch.float32, device=device
    )
    for frame_idx, output_per_frame in enumerate(outputs):
        num_detection = min(output_per_frame["scores"].size(0), NUM_MAX_DETECTION)
        scores[frame_idx, :num_detection] = output_per_frame["scores"][
            :NUM_MAX_DETECTION
        ]
        labels[frame_idx, :num_detection] = output_per_frame["labels"][
            :NUM_MAX_DETECTION
        ]
        boxes[frame_idx, :num_detection] = output_per_frame["boxes"][:NUM_MAX_DETECTION]

    return {
        "scores": scores,
        "labels": labels,
        "boxes": boxes,
    }
