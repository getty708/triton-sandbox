import json

import torch
from transformers import DetrImageProcessor

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    HUGGINGFACE_MODEL_NAME,
    HUGGINGFACE_MODEL_REVISION,
    IMAGE_TENSOR_NAME,
)
from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.wrapper.preprocess import (
    detr_preprocessing_fn,
)
from triton_benchmarks.preprocss_inference_ensemble.triton.triton_utils import (
    build_inference_response,
    convert_infer_request_to_tensors,
)


class TritonPythonModel:

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self.output_tensor_configs = model_config["output"]

        self.processor = DetrImageProcessor.from_pretrained(
            HUGGINGFACE_MODEL_NAME, revision=HUGGINGFACE_MODEL_REVISION
        )

    def execute(self, requests):
        # Prepare input tensors
        input_tensors = convert_infer_request_to_tensors(
            requests, tensor_names=[IMAGE_TENSOR_NAME]
        )

        # Preprocessing
        input_tensor = input_tensors[IMAGE_TENSOR_NAME]
        preproc_outputs: dict[str, torch.Tensor] = detr_preprocessing_fn(
            self.processor,
            input_tensor,
        )

        # Return InferenceResponse objects for each request
        response = build_inference_response(
            preproc_outputs,
            output_tensor_configs=self.output_tensor_configs,
        )
        return [response]
