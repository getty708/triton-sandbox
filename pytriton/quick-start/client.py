# import torch
import numpy as np
from pytriton.client import ModelClient

OUTPUT_TENSOR_NAME = "OUTPUT_1"


def request_inference_to_liner_model(input1_batch):
    print("=== liner ===")
    with ModelClient("localhost:8000", "linear") as client:
        result_dict = client.infer_batch(input1_batch)

    print("result_dict:", list(result_dict.keys()))
    print(f"[{OUTPUT_TENSOR_NAME}] shape:", result_dict[OUTPUT_TENSOR_NAME].shape)
    print(f"[{OUTPUT_TENSOR_NAME}] data:\n", result_dict[OUTPUT_TENSOR_NAME])


def request_inference_to_toy_model(input1_batch):
    print("=== Toy Model (multiply-by-2) ===")
    with ModelClient("localhost:8000", "multiply-by-2") as client:
        result_dict = client.infer_batch(input1_batch)

    print("result_dict:", list(result_dict.keys()))
    print(f"[{OUTPUT_TENSOR_NAME}] shape:", result_dict[OUTPUT_TENSOR_NAME].shape)
    print(f"[{OUTPUT_TENSOR_NAME}] data:\n", result_dict[OUTPUT_TENSOR_NAME])


def main():
    batch_size = 8
    input_dim = 2
    input1_batch = (
        np.arange((batch_size * input_dim))
        .reshape(batch_size, input_dim)
        .astype(np.float32)
    )
    print("[INPUT] shape:", input1_batch.shape)
    print("[INPUT] data:\n", input1_batch)

    request_inference_to_liner_model(input1_batch)
    request_inference_to_toy_model(input1_batch)


if __name__ == "__main__":
    main()
