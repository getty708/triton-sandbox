import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_benchmarks.ensemble_model_v0.dummy_model.const import (
    DUMMY_IMAGE_OUTPUT_TENSOR_NAME,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, (3, 3), stride=1, padding=(1, 1))

    def forward(self, pixel_values_input: torch.Tensor) -> dict[str, torch.Tensor]:
        pixel_values_out = F.relu(self.conv1(pixel_values_input))
        return {DUMMY_IMAGE_OUTPUT_TENSOR_NAME: pixel_values_out}
