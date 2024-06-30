import numpy as np
import torch
from loguru import logger


class Tensor:
    def __init__(self, name: str, data: np.ndarray):
        self._name = name
        self._data = data

    def name(self):
        return self._name

    def is_cpu(self):
        if isinstance(self._data, torch.Tensor) and (self._data.device.type == "cpu"):
            return False
        return True

    def as_numpy(self):
        if isinstance(self._data, torch.Tensor):
            return self._data.detach().cpu().numpy()
        return self._data

    @classmethod
    def from_dlpack(cls, name: str, dlpack_tensor: object):
        logger.warning(f"Fake Tensor.from_dlpack() is called for {name}.")
        return cls(name, torch.from_dlpack(dlpack_tensor))


class InferenceRequest:
    def __init__(
        self,
        model_name: str = "dummy_model_name",
        requested_output_names: list[str] | None = None,
        inputs: list[Tensor] | None = None,
    ):
        self._model_name = model_name
        self._requested_output_names = requested_output_names
        self._inputs = inputs
        self._index = {pb_tensor.name: i for i, pb_tensor in enumerate(inputs)}

    def inputs(self):
        return self._inputs


class InferenceResponse:
    def __init__(self, output_tensors: list[Tensor]):
        self.output_tensors = output_tensors


def get_input_tensor_by_name(inference_request, name):
    """Find an input Tensor in the inference_request that has the given
    name
    Parameters
    ----------
    inference_request : InferenceRequest
        InferenceRequest object
    name : str
        name of the input Tensor object
    Returns
    -------
    Tensor
        The input Tensor with the specified name, or None if no
        input Tensor with this name exists
    """
    input_tensors = inference_request.inputs()
    for input_tensor in input_tensors:
        if input_tensor.name() == name:
            return input_tensor

    return None


TRITON_STRING_TO_NUMPY = {
    "TYPE_BOOL": bool,
    "TYPE_UINT8": np.uint8,
    "TYPE_UINT16": np.uint16,
    "TYPE_UINT32": np.uint32,
    "TYPE_UINT64": np.uint64,
    "TYPE_INT8": np.int8,
    "TYPE_INT16": np.int16,
    "TYPE_INT32": np.int32,
    "TYPE_INT64": np.int64,
    "TYPE_FP16": np.float16,
    "TYPE_FP32": np.float32,
    "TYPE_FP64": np.float64,
    "TYPE_STRING": np.object_,
}


def triton_string_to_numpy(triton_type_string):
    return TRITON_STRING_TO_NUMPY[triton_type_string]
