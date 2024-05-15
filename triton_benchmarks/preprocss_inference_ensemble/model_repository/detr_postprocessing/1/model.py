import json

import torch
from transformers import DetrImageProcessor

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.const import (
    DETR_LOGITS_TENSOR_NAME,
    DETR_PRED_BOXES_TENSOR_NAME,
)
from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.wrapper.postprocess import (
    detr_postprocessing_fn,
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
            "facebook/detr-resnet-50", revision="no_timm"
        )

    def execute(self, requests):
        # Prepare input tensors
        model_outputs: dict[str, torch.Tensor] = convert_infer_request_to_tensors(
            requests,
            tensor_names=[DETR_LOGITS_TENSOR_NAME, DETR_PRED_BOXES_TENSOR_NAME],
        )
        postproc_outputs: dict[str, torch.Tensor] = detr_postprocessing_fn(
            self.processor,
            logits_batch=model_outputs[DETR_LOGITS_TENSOR_NAME],
            pred_boxes_batch=model_outputs[DETR_PRED_BOXES_TENSOR_NAME],
        )

        # Return InferenceResponse objects for each request
        response = build_inference_response(
            postproc_outputs,
            output_tensor_configs=self.output_tensor_configs,
        )
        return [response]
