import numpy as np
import torch
import tritonclient.grpc as grpcclient
import tritonclient.utils.cuda_shared_memory as cudashm
from loguru import logger

_DEFAULT_INPUT_TENSOR_NAME = "input"


# TODO: Separate function into two (cudashm for input and output)
def create_input_output_tensors_with_cudashm(
    batched_image: torch.Tensor,
    triton_client: grpcclient.InferenceServerClient,
    input_tensor_name: str = _DEFAULT_INPUT_TENSOR_NAME,
    output_tensor_names: list[str] = ["output_image"],
    input_dtype: str = "FP32",
    gpu_id: int = 0,
) -> tuple[list[grpcclient.InferInput], list[grpcclient.InferRequestedOutput], list]:
    input_shm_name = f"{input_tensor_name}_cudashm"
    output_shm_name = f"cudashm_for_output"

    # Create shared memory region for input and store shared memory handle
    input_byte_size = batched_image.element_size() * batched_image.numel()
    output_byte_size = input_byte_size * len(output_tensor_names)

    logger.debug(
        f"Create shared memory for input. "
        f"(tensor={input_tensor_name}, shm={input_shm_name}, size={input_byte_size})"
    )
    shm_ip_handle = cudashm.create_shared_memory_region(
        input_shm_name, input_byte_size, gpu_id
    )
    cudashm.set_shared_memory_region(
        shm_ip_handle, [np.ascontiguousarray(batched_image.cpu().numpy())]
    )
    triton_client.register_cuda_shared_memory(
        input_shm_name,
        cudashm.get_raw_handle(shm_ip_handle),
        gpu_id,
        input_byte_size,
    )

    logger.debug(
        f"Create shared memory for output. "
        f"(name={output_tensor_names}, shm={output_shm_name}, size={output_byte_size})"
    )
    shm_op_handle = cudashm.create_shared_memory_region(
        output_shm_name, output_byte_size, gpu_id
    )
    triton_client.register_cuda_shared_memory(
        output_shm_name,
        cudashm.get_raw_handle(shm_op_handle),
        gpu_id,
        output_byte_size,
    )

    # Set the parameters to use data from shared memory
    input_tensors = []
    input_tensors.append(
        grpcclient.InferInput(input_tensor_name, batched_image.shape, input_dtype)
    )
    input_tensors[-1].set_shared_memory(input_shm_name, input_byte_size)

    output_tensors = []
    for output_tensor_name in output_tensor_names:
        output_tensors.append(grpcclient.InferRequestedOutput(output_tensor_name))
        output_tensors[-1].set_shared_memory(output_shm_name, output_byte_size)

    return input_tensors, output_tensors, [shm_ip_handle, shm_op_handle]
