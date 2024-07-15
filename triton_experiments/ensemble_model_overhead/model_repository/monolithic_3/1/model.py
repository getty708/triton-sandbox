import json

import torch

from triton_experiments.ensemble_model_overhead.dummy_model.const import (
    DUMMY_IMAGE_INPUT_TENSOR_NAME,
    DUMMY_IMAGE_OUTPUT_TENSOR_NAME,
)
from triton_experiments.ensemble_model_overhead.dummy_model.model import DummyModel
from triton_experiments.triton_utils.python_backend import (
    build_inference_response,
    convert_infer_request_to_tensors,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TritonPythonModel:
    output_tensor_configs: dict | None = None
    model: torch.nn.Module | None = None

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self.output_tensor_configs = model_config["output"]

        # Init model
        self.model = DummyModel().to(device=_DEVICE)
        self.model.eval()

    def execute(self, requests):
        # Prepare input tensors
        preproc_outputs: dict[str, torch.Tensor] = convert_infer_request_to_tensors(
            requests,
            tensor_names=[DUMMY_IMAGE_INPUT_TENSOR_NAME],
            device=_DEVICE,
        )

        # Inference (1st)
        image_input = preproc_outputs[DUMMY_IMAGE_INPUT_TENSOR_NAME]
        model_outputs = self.model(image_input)
        # Inference (2nd)
        image_input = model_outputs[DUMMY_IMAGE_OUTPUT_TENSOR_NAME]
        model_outputs = self.model(image_input)
        # Inference (3rd)
        image_input = model_outputs[DUMMY_IMAGE_OUTPUT_TENSOR_NAME]
        model_outputs = self.model(image_input)

        # Return InferenceResponse objects for each request
        response = build_inference_response(
            model_outputs,
            output_tensor_configs=self.output_tensor_configs,
        )
        return [response]
