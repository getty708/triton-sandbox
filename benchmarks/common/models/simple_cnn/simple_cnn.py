import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.common.models.simple_cnn.const import SIMPLE_CNN_OUTPUT_TENSOR


class SimpleCNN(nn.Module):
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.convs = [
            nn.Conv2d(3, 3, (3, 3), stride=1, padding=(1, 1)) for _ in range(num_layers)
        ]

    def forward(self, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        x = image_tensor
        for conv in self.convs:
            x = F.relu(conv(x))
        return {SIMPLE_CNN_OUTPUT_TENSOR: x}
