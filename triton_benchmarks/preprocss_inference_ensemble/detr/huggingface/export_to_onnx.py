import time

import requests
import torch
from loguru import logger
from optimum.onnxruntime import ORTModel
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


def main():
    save_directory = "onnx/"

    model = ORTModel.from_pretrained(
        "facebook/detr-resnet-50",
        export=True,
    )
    model.save_pretrained(save_directory)


if __name__ == "__main__":
    main()
