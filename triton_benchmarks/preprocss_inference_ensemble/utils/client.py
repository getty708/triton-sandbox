from pathlib import Path

import cv2
import numpy as np
import torch
import tritonclient.grpc as grpcclient
import tritonclient.utils.cuda_shared_memory as cudashm
import tritonclient.utils.shared_memory as shm
from loguru import logger

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    DETR_LABEL_ID_TO_NAME,
    IMAGE_TENSOR_NAME,
    NUM_MAX_DETECTION,
)

MODEL_HOSTING_PATTERN_LOCAL = 0
MODEL_HOSTING_PATTERN_ALL_IN_ONE = 1
MODEL_HOSTING_PATTERN_ENSEMBPLE_PY = 10
MODEL_HOSTING_PATTERN_ENSEMBPLE_PY_PY_PY = 2
MODEL_HOSTING_PATTERN_ENSEMBPLE_PY_ONNX_PY = 3


INPUT_WITHOUT_SHM_MODE = "INPUT_WITHOUT_SHM_MODE"
INPUT_WITH_SHM_MODE = "INPUT_WITH_SHM_MODE"
INPUT_WITH_CUDASHM_MODE = "INPUT_WITH_CUDASHM_MODE"

TRITON_INPUT_IMAGE_SHM_NAME = "input_image_data"
TRITON_INPUT_IMAGE_SHM_KEY = "/input_image"

GPU_ID = 0


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


def create_batch_tensor(image: np.ndarray, batch_size: int = 1) -> torch.Tensor:
    """Create a batch tensor from the given image.

    Args:
        image: The image to create a batch tensor from.
        batch_size: The batch size to create.
    Returns:
        The batch tensor.
    """
    batch = np.stack([image] * batch_size, axis=0)
    batch_tensor = torch.from_numpy(batch).to(torch.uint8).contiguous()
    return batch_tensor


def convert_results_to_tensor_dict(
    result: dict[str, torch.Tensor] | grpcclient.InferResult
):
    if isinstance(result, grpcclient.InferResult):
        pb_result = result.get_response(as_json=False)
        tensor_names = [pb_output_tensor.name for pb_output_tensor in pb_result.outputs]
        postproc_outputs = {
            tensor_name: torch.from_numpy(result.as_numpy(tensor_name))
            for tensor_name in tensor_names
        }
    else:
        postproc_outputs = result

    return postproc_outputs


def print_output_tensor_metas(result: dict[str, torch.Tensor] | grpcclient.InferResult):
    """Print the meta information of the output tensors.

    Args:
        result: The results of the inference.
    """
    postproc_outputs = convert_results_to_tensor_dict(result)
    tensor_names = postproc_outputs.keys()

    logger.info(f"Output tensors: {tensor_names}")
    for i, tensor_name in enumerate(tensor_names):
        output_tensor = postproc_outputs[tensor_name]
        logger.info(
            f"[{i}] {tensor_name}: {output_tensor.shape} (dtype={output_tensor.dtype})"
        )


def print_detected_objects(result: dict[str, torch.Tensor] | grpcclient.InferResult):
    """Print result of the inference.

    Args:
        result: The result of the inference.
    """
    postproc_outputs = convert_results_to_tensor_dict(result)
    scores = postproc_outputs.get("scores").detach().cpu().numpy()
    labels = postproc_outputs.get("labels").detach().cpu().numpy()
    boxes = postproc_outputs.get("boxes").detach().cpu().numpy()
    if scores.ndim == 2:
        scores = scores[:, :, None]
    if labels.ndim == 2:
        labels = labels[:, :, None]

    logger.info("Detected Objects:")
    batch_size = scores.shape[0]
    for frame_idx in range(batch_size):
        logger.info(f"--- Frame {frame_idx} ---")
        for box_idx in range(NUM_MAX_DETECTION):
            _score = scores[frame_idx, box_idx, 0]
            if _score < 0:
                continue
            _label_id = int(labels[frame_idx, box_idx, 0])
            _label_name = DETR_LABEL_ID_TO_NAME[_label_id]
            _box = [int(i) for i in boxes[frame_idx, box_idx].tolist()]
            logger.info(
                (
                    f"[{box_idx}] {_label_name:<10} (ID {_label_id}) with confidence "
                    f"{_score:.3f} at location {_box}"
                )
            )


def create_input_tensors_wo_shared_memory(
    batched_image: torch.Tensor,
) -> list[grpcclient.InferInput]:
    input_tensors = [
        grpcclient.InferInput(IMAGE_TENSOR_NAME, batched_image.shape, "UINT8"),
    ]
    input_tensors[0].set_data_from_numpy(np.ascontiguousarray(batched_image.numpy()))
    return input_tensors


def create_input_tensors_with_shm(
    batched_image: torch.Tensor,
    triton_client: grpcclient.InferenceServerClient,
) -> list[grpcclient.InferInput]:
    # Create shared memory region for input and store shared memory handle
    input_byte_size = batched_image.element_size() * batched_image.numel()
    logger.info(
        f"Create shared memory for bached_image. "
        f"(name={IMAGE_TENSOR_NAME}, key={TRITON_INPUT_IMAGE_SHM_KEY}, size={input_byte_size})"
    )
    shm_ip_handle = shm.create_shared_memory_region(
        IMAGE_TENSOR_NAME,
        TRITON_INPUT_IMAGE_SHM_KEY,
        input_byte_size,
    )
    shm.set_shared_memory_region(shm_ip_handle, [batched_image.numpy()])

    # Register shared memory region for inputs with Triton Server
    triton_client.register_system_shared_memory(
        IMAGE_TENSOR_NAME,
        TRITON_INPUT_IMAGE_SHM_KEY,
        input_byte_size,
    )

    # Set the parameters to use data from shared memory
    input_tensors = []
    input_tensors.append(
        grpcclient.InferInput(IMAGE_TENSOR_NAME, batched_image.shape, "UINT8")
    )
    input_tensors[-1].set_shared_memory(IMAGE_TENSOR_NAME, input_byte_size)

    return input_tensors, shm_ip_handle


def create_input_tensors_with_cudashm(
    batched_image: torch.Tensor,
    triton_client: grpcclient.InferenceServerClient,
) -> list[grpcclient.InferInput]:
    # Create shared memory region for input and store shared memory handle
    input_byte_size = batched_image.element_size() * batched_image.numel()
    logger.info(
        f"Create shared memory for bached_image. "
        f"(name={IMAGE_TENSOR_NAME}, key={TRITON_INPUT_IMAGE_SHM_KEY}, size={input_byte_size})"
    )
    shm_ip_handle = cudashm.create_shared_memory_region(
        IMAGE_TENSOR_NAME,
        input_byte_size,
        GPU_ID,
    )
    cudashm.set_shared_memory_region(
        shm_ip_handle, [np.ascontiguousarray(batched_image.numpy())]
    )

    # Register shared memory region for inputs with Triton Server
    triton_client.register_cuda_shared_memory(
        IMAGE_TENSOR_NAME,
        cudashm.get_raw_handle(shm_ip_handle),
        GPU_ID,
        input_byte_size,
    )

    # Set the parameters to use data from shared memory
    input_tensors = []
    input_tensors.append(
        grpcclient.InferInput(IMAGE_TENSOR_NAME, batched_image.shape, "UINT8")
    )
    input_tensors[-1].set_shared_memory(IMAGE_TENSOR_NAME, input_byte_size)

    return input_tensors, shm_ip_handle
