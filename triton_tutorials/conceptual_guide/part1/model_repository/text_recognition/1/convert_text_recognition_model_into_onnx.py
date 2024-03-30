import torch
from utils.model import STRModel

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Create PyTorch Model Object
    model = STRModel(input_channels=1, output_channels=512, num_classes=37).to(_DEVICE)

    # Load model weights from external file
    state = torch.load("None-ResNet-None-CTC.pth", map_location=_DEVICE)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state)

    # Create ONNX file by tracing model
    trace_input = torch.randn(1, 1, 32, 100)
    torch.onnx.export(model, trace_input, "model.onnx", verbose=True)


if __name__ == "__main__":
    main()
