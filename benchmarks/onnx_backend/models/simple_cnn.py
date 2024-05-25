import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.onnx_backend.models.const import SIMPLE_CNN_OUTPUT_TENSOR


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, (3, 3), stride=1, padding=(1, 1))
        self.conv2 = nn.Conv2d(3, 3, (3, 3), stride=1, padding=(1, 1))
        self.conv3 = nn.Conv2d(3, 3, (3, 3), stride=1, padding=(1, 1))

    def forward(self, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.relu(self.conv1(image_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return {SIMPLE_CNN_OUTPUT_TENSOR: x}
