import torch
import tritonclient.grpc as grpcclient
from pytest_mock import MockFixture

from triton_experiments.concurrent_model_exec.client_utils import (
    create_input_output_tensors_with_cudashm,
)


def test_create_input_output_tensors_with_cudashm(mocker: MockFixture):
    batched_image = torch.rand((1, 3, 224, 224))
    triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
    input_tensor_name = "input_image"
    output_tensor_names = ["output_image_m1", "output_image_m2", "output_image_m3"]
    mocker.patch("tritonclient.utils.cuda_shared_memory.create_shared_memory_region")
    mocker.patch("tritonclient.utils.cuda_shared_memory.set_shared_memory_region")
    mocker.patch("tritonclient.utils.cuda_shared_memory.get_raw_handle")
    mocker.patch("tritonclient.grpc.InferenceServerClient.register_cuda_shared_memory")
    mocker.patch("tritonclient.grpc.InferInput.set_shared_memory")
    mocker.patch("tritonclient.grpc.InferRequestedOutput.set_shared_memory")

    input_tensors, output_tensors, shm_handles = create_input_output_tensors_with_cudashm(
        batched_image,
        triton_client,
        input_tensor_name,
        output_tensor_names=output_tensor_names,
    )

    assert isinstance(input_tensors[0], grpcclient.InferInput)
    assert len(output_tensors) == len(output_tensor_names)
    assert all(isinstance(output_tensors[i], grpcclient.InferRequestedOutput) for i in range(3))
    assert len(shm_handles) == 2
