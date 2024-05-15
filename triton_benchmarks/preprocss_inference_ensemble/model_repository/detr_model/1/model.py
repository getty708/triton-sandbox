import json

import torch
from transformers import DetrForObjectDetection

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    DETR_PIXEL_VALUES_TENSOR_NAME,
    HUGGINGFACE_MODEL_NAME,
    HUGGINGFACE_MODEL_REVISION,
)
from triton_benchmarks.preprocss_inference_ensemble.triton.triton_utils import (
    build_inference_response,
    convert_infer_request_to_tensors,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TritonPythonModel:

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self.output_tensor_configs = model_config["output"]

        self.model = DetrForObjectDetection.from_pretrained(
            HUGGINGFACE_MODEL_NAME, revision=HUGGINGFACE_MODEL_REVISION
        ).to(device=_DEVICE)

    def execute(self, requests):
        # Prepare input tensors
        preproc_outputs: dict[str, torch.Tensor] = convert_infer_request_to_tensors(
            requests,
            tensor_names=[DETR_PIXEL_VALUES_TENSOR_NAME],
        )

        # Inference
        model_outputs: dict[str, torch.Tensor] = self.model(
            pixel_values=preproc_outputs[DETR_PIXEL_VALUES_TENSOR_NAME].to(device=self.model.device),
            return_dict=True
        )

        # Return InferenceResponse objects for each request
        response = build_inference_response(
            model_outputs,
            output_tensor_configs=self.output_tensor_configs,
        )
        return [response]
