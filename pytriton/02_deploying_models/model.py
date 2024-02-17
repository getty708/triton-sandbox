import numpy as np
import torch
from pytriton.decorators import batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SampleTopyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2


SAMPLE_TOY_MODEL_NAME = "sample-toy-model"
SAMPLE_TOY_MODEL = SampleTopyModel().to(DEVICE).eval()


@batch
def infer_fn_sample_toy_model(**inputs: np.ndarray) -> list[np.ndarray]:
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to(DEVICE)
    output1_batch_tensor = SAMPLE_TOY_MODEL(input1_batch_tensor)
    output1_batch = output1_batch_tensor.cpu().detach().numpy()
    return [output1_batch]
