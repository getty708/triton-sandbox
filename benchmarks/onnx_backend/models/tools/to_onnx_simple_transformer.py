from pathlib import Path

import click
import onnx
import torch
from loguru import logger
from onnxsim import simplify

from benchmarks.onnx_backend.models.simple_transformer import SimpleTransformer

_DEFAULT_OUTPUT_DIR = Path(__file__).parent
ONNX_MODEL_FILE_NAME = "simple_transformer.onnx"
ONNX_OPTIMIZED_MODEL_FILE_NAME = "simple_transformer_optimized.onnx"


def convert_simple_transformer_to_onnx(
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    onnx_filename: Path = ONNX_MODEL_FILE_NAME,
):
    # Create a model instance and dummy outputs
    model = SimpleTransformer()
    dummy_input = torch.randn(2, 8, 128)

    logger.info(f"Converting model to ONNX.")
    # TODO: Set custom names to input/output tensors
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    onnx_program = torch.onnx.dynamo_export(
        model, dummy_input, export_options=export_options
    )
    onnx_program.save(str(output_dir / onnx_filename))
    print("Model Input:")
    print(onnx_program.model_proto.graph.input[0])


def optimize_simple_transformer_onnx_structure(
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    onnx_filename: Path = ONNX_MODEL_FILE_NAME,
    optimized_onnx_filename: Path = ONNX_OPTIMIZED_MODEL_FILE_NAME,
):
    path_input_onnx = output_dir / onnx_filename
    path_output_onnx = output_dir / optimized_onnx_filename

    # Optmization
    logger.info(f"Optmizing model graph.")
    logger.info(f"Load ONNX from {path_input_onnx}")
    model_onnx1 = onnx.load(path_input_onnx)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    logger.info(f"Export ONNX to{path_output_onnx}")
    onnx.save(model_onnx1, path_output_onnx)

    logger.info(f"Simplify ONNX graph.")
    logger.info(f"Load ONNX from {path_output_onnx}")
    model_onnx2 = onnx.load(path_output_onnx)
    model_simp, check = simplify(model_onnx2)
    logger.info(f"Export ONNX to {path_output_onnx}")
    onnx.save(model_simp, path_output_onnx)


@click.command()
@click.option("-d", "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
@click.option("--onnx_filename", type=str, default=ONNX_MODEL_FILE_NAME)
@click.option(
    "--optimized_onnx_filename",
    type=str,
    default=ONNX_OPTIMIZED_MODEL_FILE_NAME,
)
def main(
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    onnx_filename: str = ONNX_MODEL_FILE_NAME,
    optimized_onnx_filename: str = ONNX_OPTIMIZED_MODEL_FILE_NAME,
):
    convert_simple_transformer_to_onnx(output_dir, onnx_filename)
    optimize_simple_transformer_onnx_structure(
        output_dir, onnx_filename, optimized_onnx_filename
    )


if __name__ == "__main__":
    main()
