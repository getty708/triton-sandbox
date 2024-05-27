import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import tritonclient.grpc as grpcclient
import tritonclient.utils.cuda_shared_memory as cudashm
import tritonclient.utils.shared_memory as shm
import yaml
from loguru import logger
from tritonclient import utils

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DEFAULT_INPUT_TENSOR_NAME = "input"
_DEFAULT_OUTPUT_TENSOR_NAME = "output"


# =========
#  Logging
# =========
def create_logdir(args: argparse.Namespace) -> Path:
    logdir = (
        args.logdir
        / f"b{args.batch_size}-n{args.num_requests}"
        / f"{args.pipeline_architecture}"
    )
    logdir.mkdir(parents=True, exist_ok=True)
    return logdir


# ==============
#  Input Tensor
# ==============
def load_image(image_path: Path) -> np.ndarray:
    """Load image data from the given path.

    Args:
        image_path: path to the input image
    Returns:
        np.array with shape=(CHANNELS=3, HEIGHT=1080, WIDTH=1920)
    """
    image_data = cv2.imread(str(image_path))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    image_data = image_data.transpose(2, 0, 1)
    return image_data


def create_batch_tensor(
    image: np.ndarray,
    batch_size: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a batch tensor from the given image.

    Args:
        image: The image to create a batch tensor from.
        batch_size: The batch size to create.
    Returns:
        The batch tensor.
    """
    batch = np.stack([image] * batch_size, axis=0)
    batch_tensor = torch.from_numpy(batch).to(dtype=dtype).contiguous()
    if device is not None:
        batch_tensor = batch_tensor.to(device=device)
    return batch_tensor


# TODO: Move to dummy_model/helper.py
def prepare_image_batch_tensor(
    input_image_path: Path,
    batch_size: int,
) -> torch.Tensor:
    # Load image data.
    image_data = load_image(input_image_path)
    logger.info(f"Input Image: {image_data.shape} ({image_data.dtype})")
    image_batch = create_batch_tensor(
        image_data, batch_size=batch_size, device=torch.device("cpu")
    )
    logger.info(f"Input Tensor: {image_batch.shape} ({image_batch.dtype})")

    return image_batch


# ============================
#  Preparation of InferInputs
# ============================


def create_input_tensors_wo_shared_memory(
    batched_image: torch.Tensor,
    tensor_name: str = _DEFAULT_INPUT_TENSOR_NAME,
) -> list[grpcclient.InferInput]:
    input_tensors = [
        grpcclient.InferInput(tensor_name, batched_image.shape, "FP32"),
    ]
    batched_image_np = np.ascontiguousarray(batched_image.numpy().astype(np.float32))
    input_tensors[0].set_data_from_numpy(batched_image_np)
    return input_tensors


def create_input_output_tensors_with_cudashm(
    batched_image: torch.Tensor,
    triton_client: grpcclient.InferenceServerClient,
    input_tensor_name: str = _DEFAULT_INPUT_TENSOR_NAME,
    output_tensor_name: str = _DEFAULT_OUTPUT_TENSOR_NAME,
    input_dtype: str = "FP32",
    gpu_id: int = 0,
) -> tuple[list[grpcclient.InferInput], list[grpcclient.InferRequestedOutput], list]:
    input_shm_name = f"{input_tensor_name}_cudashm"
    output_shm_name = f"{output_tensor_name}_cudashm"

    # Create shared memory region for input and store shared memory handle
    input_byte_size = batched_image.element_size() * batched_image.numel()
    output_byte_size = input_byte_size

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
        f"(name={output_tensor_name}, shm={output_shm_name}, size={output_byte_size})"
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
    output_tensors.append(grpcclient.InferRequestedOutput(output_tensor_name))
    output_tensors[-1].set_shared_memory(output_shm_name, output_byte_size)

    return input_tensors, output_tensors, [shm_ip_handle, shm_op_handle]


def cleanup_shared_memory(
    triton_client: grpcclient.InferenceServerClient,
    shm_handles: list | None = None,
):
    if shm_handles is None:
        return

    triton_client.unregister_system_shared_memory()
    triton_client.unregister_cuda_shared_memory()
    for shm_handle in shm_handles:
        if _DEVICE == "cpu":
            shm.destroy_shared_memory_region(shm_handle)
        elif _DEVICE == "cuda":
            cudashm.destroy_shared_memory_region(shm_handle)


# =====================
#  Helper Functions
# =====================


def convert_results_on_cudashm_to_tensor_dict(
    result: grpcclient.InferResult,
    shm_output_handle: Any,
    output_tensor_name: str = _DEFAULT_OUTPUT_TENSOR_NAME,
):
    if isinstance(result, grpcclient.InferResult):
        # TODO: Create torch.Tensor from CUDA shared memory.
        output0 = result.get_output(output_tensor_name)
        if output0 is None:
            raise ValueError(f"Output tensor '{output_tensor_name}' not found.")
        output_tensor_np = cudashm.get_contents_as_numpy(
            shm_output_handle, utils.triton_to_np_dtype(output0.datatype), output0.shape
        )
        postproc_outputs = {
            output_tensor_name: torch.from_numpy(output_tensor_np),
        }
    else:
        postproc_outputs = result

    return postproc_outputs


def print_output_tensor_metas(output_tensors: dict[str, torch.Tensor]):
    """Print the meta information of the output tensors.

    Args:
        result: The results of the inference.
    """
    tensor_names = output_tensors.keys()

    logger.info(f"Output tensors: {tensor_names}")
    for i, tensor_name in enumerate(tensor_names):
        output_tensor = output_tensors[tensor_name]
        logger.info(
            f"[{i}] {tensor_name}: {output_tensor.shape} "
            f"(dtype={output_tensor.dtype}, device={output_tensor.device})"
        )


def save_client_stats(logdir: Path, df_stats: pd.DataFrame):
    csv_path = logdir / "client_stats.csv"
    logger.info(f"Save client stats to {csv_path}")
    df_stats.to_csv(csv_path, index=False)


def save_triton_inference_stats(logdir: Path, client: grpcclient.InferenceServerClient):
    stats: dict = client.get_inference_statistics(as_json=True)
    yaml_path = logdir / "triton_stats.yaml"
    logger.info(f"Save stats to {yaml_path}")
    with open(yaml_path, "w") as f:
        yaml.dump(stats, f)
