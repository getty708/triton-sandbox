import pandas as pd
import torch
from loguru import logger
from torch.utils.dlpack import from_dlpack, to_dlpack

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    NUM_MAX_DETECTION,
)

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    import triton_benchmarks.preprocss_inference_ensemble.triton.tests.fake_triton_python_backend_utils as pb_utils

MODEL_OUTPUT_NAME_KEY_NAME = "name"
MODEL_OUTPUT_DIMS_KEY_NAME = "dims"
MODEL_OUTPUT_DATA_TYPE_KEY_NAME = "data_type"

# Alias for c_python_backend_utils.InferenceRequest
InferenceRequestType = object
InferenceResponseType = object


def extract_output_dtypes(
    output_tensor_names: list[str], model_config: dict
) -> dict[str, dict]:
    output_dtypes = {}
    for tensor_name in output_tensor_names:
        output_tensor_config = pb_utils.get_output_config_by_name(
            model_config, tensor_name
        )
        output_dtypes[tensor_name] = pb_utils.triton_string_to_numpy(
            output_tensor_config.get(MODEL_OUTPUT_DATA_TYPE_KEY_NAME)
        )
    logger.info(f"output_dtypes: {output_dtypes}")
    return output_dtypes


def convert_infer_request_to_tensors(
    requests: InferenceRequestType,
    tensor_names: list[str] = ["image"],
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
            logger.warning(
                f"Using as_numpy() for input tensor conversion (name={tensor_name})."
            )
            tensor = torch.from_numpy(pb_tensor.as_numpy())
        output_tensors[tensor_name] = tensor
    return output_tensors


def get_tensor_dims(tensor_meta: dict, fill_second_dim: bool = False):
    tensor_dims = tensor_meta[MODEL_OUTPUT_DIMS_KEY_NAME]
    if fill_second_dim and len(tensor_dims) >= 2:
        if tensor_dims[1] == -1:
            tensor_dims[1] = NUM_MAX_DETECTION
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
        pb_output_tensor = pb_utils.Tensor.from_dlpack(
            tensor_name, to_dlpack(pytorch_tensor.contiguous())
        )
        output_tensors.append(pb_output_tensor)

    inference_response = pb_utils.InferenceResponse(output_tensors=output_tensors)
    return inference_response


def convert_triton_stats_to_dataframe(triton_stats_dict: dict) -> pd.DataFrame:
    df = pd.json_normalize(
        triton_stats_dict["model_stats"],
    )
    df.drop("batch_stats", axis=1, inplace=True)
    print(df)
    df = df.set_index("name").fillna("0").astype(int)
    df.index.name = "model"
    return df


def summarize_triton_stats(df_stats: pd.DataFrame) -> pd.DataFrame:
    proc_names = [
        "success",
        "queue",
        "compute_input",
        "compute_infer",
        "compute_output",
    ]

    df_summary = []
    for proc_name in proc_names:
        col_name_count = f"inference_stats.{proc_name}.count"
        col_name_ns = f"inference_stats.{proc_name}.ns"
        if col_name_count not in df_stats.columns:
            continue
        df_metric = df_stats[col_name_ns] / df_stats[col_name_count]
        df_metric = df_metric / 1e6  # Convert ns to ms
        df_summary.append(df_metric.to_frame(f"{proc_name}.mean_ms"))

    df_summary = pd.concat(df_summary, axis=1)
    return df_summary
