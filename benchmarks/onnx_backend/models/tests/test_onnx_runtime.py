"""Run model with onnx runtime.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from benchmarks.onnx_backend.models.const import (
    SIMPLE_TRANSFORMER_EMBEDDING_DIM,
    SIMPLE_TRANSFORMER_SRC_SRQUENCE_LEN,
)

_ONNX_FILE_DIR = Path(__file__).parents[1] / "tools"


@pytest.mark.parametrize(
    "batch_size",
    (1, 8, 32),
)
def test_run_transforner_on_onnxruntime(batch_size: int):
    # Load ONNX model
    onnx_path = _ONNX_FILE_DIR / "simple_transformer_optimized.onnx"
    assert onnx_path.exists()
    ort_session = ort.InferenceSession(onnx_path)
    # Prepare Inputs
    input_tensor = np.random.randn(
        batch_size,
        SIMPLE_TRANSFORMER_SRC_SRQUENCE_LEN,
        SIMPLE_TRANSFORMER_EMBEDDING_DIM,
    ).astype(np.float32)
    inputs = {ort_session.get_inputs()[0].name: input_tensor}

    # Run inference
    outputs = ort_session.run(None, inputs)

    # Check outputs
    print(outputs[0])
    print(outputs[0].shape)
    assert list(outputs[0].shape) == [
        batch_size,
        SIMPLE_TRANSFORMER_SRC_SRQUENCE_LEN,
        SIMPLE_TRANSFORMER_EMBEDDING_DIM,
    ]
