from pathlib import Path

import click
import onnx
import torch
from loguru import logger
from onnxsim import simplify

from benchmarks.common.models.simple_cnn.const import (
    SIMPLE_CNN_INPUT_TENSOR,
    SIMPLE_CNN_OUTPUT_TENSOR,
)
from benchmarks.common.models.simple_cnn.simple_cnn import SimpleCNN

DEFAULT_ONNX_MODEL_FILE_NAME = "simple_cnn.onnx"
DEFAULT_SIMPLE_CNN_ONNX_PATH = (
    Path(__file__).parent / "outputs" / DEFAULT_ONNX_MODEL_FILE_NAME
)


# def convert_pytorch_model_to_onnx(
#     onnx_path: Path = DEFAULT_SIMPLE_CNN_ONNX_PATH,
#     num_layers: int = 3,
# ):
#     """Export SimpleCNN model to ONNX with Dynamo."""
#     # Create a model instance and dummy outputs
#     batch_size = 8
#     model = SimpleCNN(num_layers=num_layers)
#     model.eval()
#     dummy_input = torch.randn(batch_size, 3, 1080, 1920).detach()

#     logger.info(f"Converting model to ONNX.")
#     logger.info(f"input.shape={dummy_input.shape}")
#     torch.onnx.export(
#         model=model,
#         args=(dummy_input),
#         f=onnx_path,
#         opset_version=17,
#         input_names=[SIMPLE_CNN_INPUT_TENSOR],
#         output_names=[SIMPLE_CNN_OUTPUT_TENSOR],
#         # FIXME: Dynamic batching raise RuntimeError.
#         dynamic_axes={
#             "input": {0: "batch_size"},
#             "output": {0: "batch_size"},
#         },
#     )


def convert_pytorch_model_to_onnx(
    onnx_path: Path = DEFAULT_SIMPLE_CNN_ONNX_PATH,
    num_layers: int = 3,
):
    """Export SimpleCNN model to ONNX with Dynamo."""
    # Create a model instance and dummy outputs
    batch_size = 8
    model = SimpleCNN(num_layers=num_layers)
    dummy_input = torch.randn(batch_size, 3, 1080, 1920)

    # Convert to ONNX with Dynamo.
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    onnx_program = torch.onnx.dynamo_export(
        model, dummy_input, export_options=export_options
    )
    onnx_program.save(str(onnx_path))


def optimize_simple_transformer_onnx_structure(
    onnx_path: Path = DEFAULT_SIMPLE_CNN_ONNX_PATH,
):
    # Optmization
    logger.info(f"Optmizing model graph.")
    logger.info(f"Load ONNX from {onnx_path}")
    model_onnx1 = onnx.load(onnx_path)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    logger.info(f"Export ONNX to{onnx_path}")
    onnx.save(model_onnx1, onnx_path)

    logger.info(f"Simplify ONNX graph.")
    logger.info(f"Load ONNX from {onnx_path}")
    model_onnx2 = onnx.load(onnx_path)
    model_simp, check = simplify(model_onnx2)
    logger.info(f"Export ONNX to {onnx_path}")
    onnx.save(model_simp, onnx_path)


@click.command()
@click.option("-o", "--onnx-path", type=Path, default=DEFAULT_SIMPLE_CNN_ONNX_PATH)
@click.option("-l", "--num-layers", type=int, default=3)
def main(
    onnx_path: Path = DEFAULT_SIMPLE_CNN_ONNX_PATH,
    num_layers: int = 3,
):
    convert_pytorch_model_to_onnx(onnx_path, num_layers=num_layers)
    optimize_simple_transformer_onnx_structure(onnx_path)


if __name__ == "__main__":
    main()
