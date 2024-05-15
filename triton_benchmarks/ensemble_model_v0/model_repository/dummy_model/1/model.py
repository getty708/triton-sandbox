import json

import nvtx
import torch

from triton_benchmarks.ensemble_model_v0.dummy_model.const import \
    DUMMY_IMAGE_INPUT_TENSOR_NAME
from triton_benchmarks.ensemble_model_v0.dummy_model.wrapper import DummyModel
from triton_benchmarks.triton_utils.python_backend import (
    build_inference_response, convert_infer_request_to_tensors)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TritonPythonModel:

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self.output_tensor_configs = model_config["output"]

        # Init model
        self.model = DummyModel().to(device=_DEVICE)
        self.model.eval()

    @nvtx.annotate(message="triton_execute")
    def execute(self, requests):
        # Prepare input tensors
        with nvtx.annotate(message="prepare_inputs"):
            preproc_outputs: dict[str, torch.Tensor] = convert_infer_request_to_tensors(
                requests,
                tensor_names=[DUMMY_IMAGE_INPUT_TENSOR_NAME],
                device=_DEVICE,
            )

        # Inference
        with nvtx.annotate(message="inference"):
            model_outputs: dict[str, torch.Tensor] = self.model(
                pixel_values_input=preproc_outputs[DUMMY_IMAGE_INPUT_TENSOR_NAME]
            )

        # Return InferenceResponse objects for each request
        with nvtx.annotate(message="prepare_outputs"):
            response = build_inference_response(
                model_outputs,
                output_tensor_configs=self.output_tensor_configs,
            )
        return [response]
