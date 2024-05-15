import json

import torch

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    IMAGE_TENSOR_NAME,
)
from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.wrapper.detection_service import (
    detr_end2end_inference,
    init_detr_model,
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

        # Init model
        self.model, self.processor = (
            init_detr_model()
        )  # model: DetrForObjectDetection, processor: DetrImageProcessor

    def execute(self, requests):
        # Prepare input tensors
        input_tensors = convert_infer_request_to_tensors(
            requests, tensor_names=[IMAGE_TENSOR_NAME]
        )

        # Inference
        image_batch = input_tensors[IMAGE_TENSOR_NAME]
        postproc_outputs = detr_end2end_inference(
            self.model, self.processor, image_batch
        )

        # Return InferenceResponse objects for each request
        response = build_inference_response(
            postproc_outputs,
            output_tensor_configs=self.output_tensor_configs,
        )
        return [response]
