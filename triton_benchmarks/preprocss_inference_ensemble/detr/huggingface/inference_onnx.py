from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from loguru import logger
from PIL import Image
from transformers import DetrImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput


def get_dummy_input(batch_size: int) -> np.ndarray:
    return np.random.randn(batch_size, 3, 750, 1333).astype(np.float32)


def get_sammple_image(batch_size: int) -> np.ndarray:
    img_path = (
        Path(__file__).parents[2]
        / "samples"
        / "pexels-anna-tarazevich-14751175-fullhd.jpg"
    )
    img = Image.open(img_path)
    logger.info(f"Load a sample image from {img_path}")

    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    inputs: BatchFeature = processor(images=img, return_tensors="pt")
    input_tensor = inputs.data["pixel_values"]
    logger.info(
        f"Preprocessed image: {input_tensor.shape} (dtype={input_tensor.dtype})"
    )
    input_np = input_tensor.numpy()
    input_np = np.tile(input_np, (batch_size, 1, 1, 1))
    logger.info(f"Batched image: {input_np.shape} (dtype={input_np.dtype})")
    return input_np


def main():
    # Load the ONNX model
    model_path = Path("onnx", "model.onnx")
    session = ort.InferenceSession(model_path, provider=["CPUExecutionProvider"])
    logger.info(f"Model loaded from {model_path}")
    print(ort.get_device())
    print(session.get_providers())
    logger.info("Inputs:")
    for i, input in enumerate(session.get_inputs()):
        logger.info(f"  [{i}] {input.name} (shape={input.shape}, type={input.type})")
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )

    # Prepare dummy data
    batch_size = 4
    input_name = session.get_inputs()[0].name
    # input_data = get_dummy_input(batch_size)
    input_data = get_sammple_image(batch_size)

    # Inference
    outputs = session.run(None, {input_name: input_data})

    # Show the output shapes
    print("Output shapes:")
    for i in range(len(outputs)):
        print(f"- {i}: {outputs[i].shape}")

    # == Post Processing ==
    # target_sizes = torch.tensor([input_data.size[::-1]])
    target_sizes = torch.tensor([[1080, 1920]] * batch_size)
    det_outputs = DetrObjectDetectionOutput(
        logits=torch.tensor(outputs[0]),
        pred_boxes=torch.tensor(outputs[1]),
    )
    results = processor.post_process_object_detection(
        det_outputs, target_sizes=target_sizes, threshold=0.9
    )

    for batch_idx in range(batch_size):
        print(f"== Batch {batch_idx} ==")
        for score, label, box in zip(
            results[batch_idx]["scores"],
            results[batch_idx]["labels"],
            results[batch_idx]["boxes"],
        ):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {label.item()} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )


if __name__ == "__main__":
    main()
