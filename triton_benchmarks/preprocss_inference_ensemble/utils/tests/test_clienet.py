from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm
from loguru import logger

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    IMAGE_TENSOR_NAME,
)
from triton_benchmarks.preprocss_inference_ensemble.utils.client import (
    create_input_tensors_with_shm,
    create_input_tensors_wo_shared_memory,
)


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def batched_image_shape(batch_size: int) -> tuple[int, int, int, int]:
    return (batch_size, 3, 1080, 1920)


@pytest.fixture
def batched_image(batched_image_shape: tuple[int, int, int, int]) -> torch.Tensor:
    input_tensor = torch.randint(0, 255, size=batched_image_shape, dtype=torch.uint8)
    return input_tensor


@pytest.fixture
def mock_triton_client():
    # Create a mock of grpcclient.InferenceServerClient() and it's method.
    mock_client = MagicMock(spec=grpcclient.InferenceServerClient)
    mock_client.register_system_shared_memory.return_value = None
    return mock_client


def test_create_input_tensors_wo_shared_memory(
    batched_image_shape: tuple[int, int, int, int], batched_image: torch.Tensor
):
    infer_input_tensors = create_input_tensors_wo_shared_memory(batched_image)

    assert len(infer_input_tensors) == 1
    assert infer_input_tensors[0].name() == IMAGE_TENSOR_NAME
    assert infer_input_tensors[0].datatype() == "UINT8"
    np.testing.assert_array_equal(infer_input_tensors[0].shape(), batched_image_shape)


def test_create_input_tensors_with_shm(
    mock_triton_client: MagicMock,
    batched_image_shape: tuple[int, int, int, int],
    batched_image: torch.Tensor,
):
    infer_input_tensors, shm_ip_handle = create_input_tensors_with_shm(
        batched_image,
        mock_triton_client,
    )

    # Check input image tensor metadata.
    assert len(infer_input_tensors) == 1
    assert infer_input_tensors[0].name() == IMAGE_TENSOR_NAME
    assert infer_input_tensors[0].datatype() == "UINT8"
    np.testing.assert_array_equal(infer_input_tensors[0].shape(), batched_image_shape)
    # Checl shared memory status and clean up.
    mock_triton_client.register_system_shared_memory.assert_called_once()
    assert len(shm.mapped_shared_memory_regions()) == 1
    shm.destroy_shared_memory_region(shm_ip_handle)
    assert len(shm.mapped_shared_memory_regions()) == 0
