import torch
from loguru import logger
from torch.utils.dlpack import from_dlpack, to_dlpack

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    import benchmarks.triton_utils.tests.fake_triton_python_backend_utils as pb_utils

MODEL_OUTPUT_NAME_KEY_NAME = "name"
MODEL_OUTPUT_DIMS_KEY_NAME = "dims"
MODEL_OUTPUT_DATA_TYPE_KEY_NAME = "data_type"

# Alias for c_python_backend_utils.InferenceRequest
InferenceRequestType = object
InferenceResponseType = object


def extract_output_dtypes(output_tensor_names: list[str], model_config: dict) -> dict[str, dict]:
    output_dtypes = {}
    for tensor_name in output_tensor_names:
        output_tensor_config = pb_utils.get_output_config_by_name(model_config, tensor_name)
        output_dtypes[tensor_name] = pb_utils.triton_string_to_numpy(
            output_tensor_config.get(MODEL_OUTPUT_DATA_TYPE_KEY_NAME)
        )
    logger.info(f"output_dtypes: {output_dtypes}")
    return output_dtypes


def convert_infer_request_to_tensors(
    requests: InferenceRequestType,
    tensor_names: list[str] = ["image"],
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    if len(requests) > 1:
        raise NotImplementedError(
            f"Dynamic batching is not supported. len(requests)={len(requests)}."
        )

    request = requests[0]
    output_tensors = {}
    for tensor_name in tensor_names:
        pb_tensor = pb_utils.get_input_tensor_by_name(request, tensor_name)
        # TODO: Compare performance improvements by using dlpack.
        if hasattr(pb_tensor, "to_dlpack"):
            tensor = from_dlpack(pb_tensor.to_dlpack())
        else:
            logger.warning(f"Using as_numpy() for input tensor conversion (name={tensor_name}).")
            tensor = torch.from_numpy(pb_tensor.as_numpy())
        if (device == "cuda") and (not tensor.is_cuda):
            logger.warning(
                f"Expected tensor location is {device}, but it's on {tensor.device}. Moving tensor to {device}."
            )
            tensor = tensor.to(device)
        output_tensors[tensor_name] = tensor
    return output_tensors


def get_tensor_dims(tensor_meta: dict, fill_second_dim: bool = False, num_max_bbox: int = 100):
    tensor_dims = tensor_meta[MODEL_OUTPUT_DIMS_KEY_NAME]
    if fill_second_dim and len(tensor_dims) >= 2:
        if tensor_dims[1] == -1:
            tensor_dims[1] = num_max_bbox
    return tensor_dims


def build_inference_response(
    tensor_dict: dict[str, torch.Tensor], output_tensor_configs: list[dict]
) -> InferenceResponseType:
    output_tensors = []
    for tensor_meta in output_tensor_configs:
        tensor_name = tensor_meta[MODEL_OUTPUT_NAME_KEY_NAME]
        tensor_dtype_str = tensor_meta[MODEL_OUTPUT_DATA_TYPE_KEY_NAME]
        tensor_dims = get_tensor_dims(tensor_meta, fill_second_dim=True)

        pytorch_tensor = tensor_dict[tensor_name].view(tensor_dims)
        if tensor_dtype_str == "TYPE_FP32":
            pytorch_tensor = pytorch_tensor.to(dtype=torch.float32)
        elif tensor_dtype_str == "TYPE_UINT8":
            pytorch_tensor = pytorch_tensor.to(dtype=torch.uint8)
        elif tensor_dtype_str == "TYPE_INT64":
            pytorch_tensor = pytorch_tensor.to(dtype=torch.int64)
        else:
            logger.warning(
                f"Given dtype ({tensor_dtype_str}) is not supported. Skipping dtype conversion."
            )
        pb_output_tensor = pb_utils.Tensor.from_dlpack(tensor_name, to_dlpack(pytorch_tensor))
        output_tensors.append(pb_output_tensor)

    inference_response = pb_utils.InferenceResponse(output_tensors=output_tensors)
    return inference_response
