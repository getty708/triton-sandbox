import time
from pathlib import Path

import requests
import torch
from loguru import logger
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers.image_processing_utils import BatchFeature


def main():
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    img_path = (
        Path(__file__).parents[2]
        / "samples"
        / "pexels-anna-tarazevich-14751175-fullhd.jpg"
    )
    image = Image.open(img_path)

    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm"
    )

    start = time.time()
    inputs: BatchFeature = processor(images=image, return_tensors="pt")
    ts_preproc = time.time()
    outputs = model(**inputs)
    ts_inference = time.time()

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]
    ts_postproc = time.time()

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    logger.info("Preprocessing time  : {:.2f} ms", (ts_preproc - start) * 1000)
    logger.info("Inference time      : {:.2f} ms", (ts_inference - ts_preproc) * 1000)
    logger.info("Postprocessing time : {:.2f} ms", (ts_postproc - ts_inference) * 1000)

    # Check the inputs
    logger.info(f"Model Inputs: {inputs.data.keys()} (inputs={type(inputs)})")
    for i, key in enumerate(inputs.data.keys()):
        tensor = inputs.data[key]
        logger.info(f"({i}) {key}: {tensor.shape} ({tensor.dtype})")
    # Check the inputs
    logger.info(f"Model Outputs: {type(outputs)}")
    attr_list = [
        "logits",
        "pred_boxes",
        "last_hidden_state",
        "encoder_last_hidden_state",
    ]
    for i, key in enumerate(attr_list):
        tensor = getattr(outputs, key)
        logger.info(f"- {key}: {tensor.shape} ({tensor.dtype})")


if __name__ == "__main__":
    main()
