from pathlib import Path

import click
import torch

from benchmarks.onnx_backend.models.simple_cnn import SimpleCNN

ONNX_MODEL_FILE_NAME = "simple_cnn.onnx"
DEFAULT_SIMPLE_CNN_ONNX_PATH = Path(__file__).parent / ONNX_MODEL_FILE_NAME


@click.command()
@click.option("-o", "--onnx_file", type=Path, default=DEFAULT_SIMPLE_CNN_ONNX_PATH)
def convert_simple_cnn_to_onnx(
    onnx_file: Path = DEFAULT_SIMPLE_CNN_ONNX_PATH,
):
    # Create a model instance and dummy outputs
    model = SimpleCNN()
    dummy_input = torch.randn(8, 3, 1080, 1920)

    # Convert to ONNX
    # TODO: Set custom names to input/output tensors
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    onnx_program = torch.onnx.dynamo_export(
        model, dummy_input, export_options=export_options
    )
    onnx_program.save(str(onnx_file))
    print("Model Input:")
    print(onnx_program.model_proto.graph.input[0])


if __name__ == "__main__":
    convert_simple_cnn_to_onnx()
