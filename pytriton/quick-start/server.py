#!/usr/bin/python
import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# == Initialize Models ==

LINEAR_MODEL_NAME = "linear"
LINEAR_MODEL_IN_DIM = 2
LINEAR_MODEL_OUT_DIM = 3
MULTIPLY_BY_2_MODEL_NAME = "multiply-by-2"


class MultiplyBy2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2


LINEAR_MODEL = (
    torch.nn.Linear(LINEAR_MODEL_IN_DIM, LINEAR_MODEL_OUT_DIM).to(DEVICE).eval()
)
MULTIPLY_BY_2_MODEL = MultiplyBy2Model().to(DEVICE).eval()


# == Define Inference Functions ==
@batch
def infer_fn_linear(**inputs: np.ndarray) -> list[np.ndarray]:
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to(DEVICE)
    output1_batch_tensor = LINEAR_MODEL(input1_batch_tensor)
    output1_batch = output1_batch_tensor.cpu().detach().numpy()
    return [output1_batch]


@batch
def infer_fn_multiply_by_2(**inputs: np.ndarray) -> list[np.ndarray]:
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to(DEVICE)
    output1_batch_tensor = MULTIPLY_BY_2_MODEL(input1_batch_tensor)
    output1_batch = output1_batch_tensor.cpu().detach().numpy()
    return [output1_batch]


# == Main ==


def main():
    with Triton() as triton:
        # Bind 2 models
        triton.bind(
            model_name=LINEAR_MODEL_NAME,
            infer_func=infer_fn_linear,
            inputs=[
                Tensor(dtype=np.float32, shape=(LINEAR_MODEL_IN_DIM,)),
            ],
            outputs=[
                Tensor(dtype=np.float32, shape=(LINEAR_MODEL_OUT_DIM,)),
            ],
            config=ModelConfig(max_batch_size=128),
        )
        triton.bind(
            model_name=MULTIPLY_BY_2_MODEL_NAME,
            infer_func=infer_fn_multiply_by_2,
            inputs=[
                Tensor(dtype=np.float32, shape=(-1,)),
            ],
            outputs=[
                Tensor(dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128),
        )

        # Serve two models
        triton.serve()


if __name__ == "__main__":
    main()
