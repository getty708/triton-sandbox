from pathlib import Path

import torch
from benchmarks.ensemble_model_overhead.dummy_model.const import (
    DUMMY_IMAGE_OUTPUT_TENSOR_NAME,
)
from benchmarks.ensemble_model_overhead.dummy_model.model import DummyModel
from benchmarks.ensemble_model_overhead.utils.client import prepare_image_batch_tensor
from loguru import logger

_REPO_ROOT = Path(__file__).parents[4]
SAMPLE_IMAGE_PATH = _REPO_ROOT / "data" / "pexels-anna-tarazevich-14751175-fullhd.jpg"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_dummy_model():
    batch_size = 2
    pixel_value = torch.rand((batch_size, 3, 1080, 1920)).to(device=_DEVICE)

    model = DummyModel().to(device=torch.device(_DEVICE))
    output = model(pixel_value)
    pixel_value = output[DUMMY_IMAGE_OUTPUT_TENSOR_NAME]

    assert isinstance(model, torch.nn.Module)
    assert list(pixel_value.size()) == [batch_size, 3, 1080, 1920]
    assert pixel_value.device == torch.device(_DEVICE)


def test_dummy_model_repetitive_inference():
    batch_size = 8
    image_input = prepare_image_batch_tensor(SAMPLE_IMAGE_PATH, batch_size)
    image_input = image_input.to(dtype=torch.float32, device=_DEVICE)
    num_requests = 100
    model = DummyModel().to(device=torch.device(_DEVICE))

    success = 0
    for i in range(num_requests):
        output = model(image_input)
        image_out = output[DUMMY_IMAGE_OUTPUT_TENSOR_NAME]
        logger.debug(f"Request[{i}] {image_out.size()}")
        success += 1

    assert success == num_requests
