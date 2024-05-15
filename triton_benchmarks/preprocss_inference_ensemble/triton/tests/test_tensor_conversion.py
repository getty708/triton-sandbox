import numpy as np
import torch

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    # Use fake module for testing
    import triton_benchmarks.preprocss_inference_ensemble.triton.tests.fake_triton_python_backend_utils as pb_utils

from triton_benchmarks.preprocss_inference_ensemble.triton.triton_utils import (
    build_inference_response,
    convert_infer_request_to_tensors,
)


def test_convert_infer_request_to_tensors():
    batch_size = 2
    batch_shape = (batch_size, 3, 1080, 1920)
    tensor_name = "image"
    dummy_image = np.random.randint(0, 255, batch_shape, dtype=np.uint8)
    request = pb_utils.InferenceRequest(
        model_name="dummy_model_name",
        requested_output_names=[tensor_name],
        inputs=[pb_utils.Tensor(tensor_name, dummy_image)],
    )

    # convert InferenceRequest to tensors
    output_tensors = convert_infer_request_to_tensors(
        [request], tensor_names=[tensor_name]
    )

    # check the output
    assert set(output_tensors.keys()) == set([tensor_name])
    assert isinstance(output_tensors[tensor_name], torch.Tensor)
    np.testing.assert_array_equal(output_tensors[tensor_name].size(), batch_shape)


def test_build_inference_response_no_reshape():
    batch_size = 2
    tensor_dict = {
        "pixel_values": torch.rand(
            size=[batch_size, 3, 750, 1333], dtype=torch.float32
        ),
        "pixel_mask": torch.randint(
            0, 255, size=[batch_size, 750, 1333], dtype=torch.int64
        ),
    }
    output_tensor_configs = [
        {
            "name": "pixel_values",
            "data_type": "TYPE_FP32",
            "dims": [-1, 3, 750, 1333],
            "label_filename": "",
            "is_shape_tensor": False,
        },
        {
            "name": "pixel_mask",
            "data_type": "TYPE_INT64",
            "dims": [-1, 750, 1333],
            "label_filename": "",
            "is_shape_tensor": False,
        },
    ]

    response = build_inference_response(tensor_dict, output_tensor_configs)

    for pb_tensor in response.output_tensors:
        if pb_tensor.name() == "pixel_values":
            np.testing.assert_array_equal(
                pb_tensor.as_numpy().shape, (batch_size, 3, 750, 1333)
            )
            assert pb_tensor.as_numpy().dtype == np.float32
        elif pb_tensor.name() == "pixel_mask":
            np.testing.assert_array_equal(
                pb_tensor.as_numpy().shape, (batch_size, 750, 1333)
            )
            assert pb_tensor.as_numpy().dtype == np.int64


def test_build_inference_response_with_reshape():
    batch_size = 2
    num_queries = 100
    tensor_dict = {
        "scores": torch.rand(size=[batch_size, num_queries], dtype=torch.float32),
        "labels": torch.rand(size=[batch_size, num_queries], dtype=torch.float32),
        "boxes": torch.randint(
            0, 100, size=[batch_size, num_queries, 4], dtype=torch.float32
        ),
    }
    output_tensor_configs = [
        {
            "name": "scores",
            "data_type": "TYPE_FP32",
            "dims": [-1, num_queries, 1],
            "label_filename": "",
            "is_shape_tensor": False,
        },
        {
            "name": "labels",
            "data_type": "TYPE_INT64",
            "dims": [-1, num_queries, 1],
            "label_filename": "",
            "is_shape_tensor": False,
        },
        {
            "name": "boxes",
            "data_type": "TYPE_FP32",
            "dims": [-1, num_queries, 4],
            "label_filename": "",
            "is_shape_tensor": False,
        },
    ]

    response = build_inference_response(tensor_dict, output_tensor_configs)

    for pb_tensor in response.output_tensors:
        if pb_tensor.name() == "scores":
            np.testing.assert_array_equal(
                pb_tensor.as_numpy().shape, (batch_size, num_queries, 1)
            )
            assert pb_tensor.as_numpy().dtype == np.float32
        elif pb_tensor.name() == "labels":
            np.testing.assert_array_equal(
                pb_tensor.as_numpy().shape, (batch_size, num_queries, 1)
            )
            assert pb_tensor.as_numpy().dtype == np.int64
        elif pb_tensor.name() == "boxes":
            np.testing.assert_array_equal(
                pb_tensor.as_numpy().shape, (batch_size, num_queries, 4)
            )
            assert pb_tensor.as_numpy().dtype == np.float32


def test_build_inference_response_with_two_dynamic_dim():
    batch_size = 2
    num_queries = 100
    tensor_dict = {
        "scores": torch.rand(size=[batch_size, num_queries], dtype=torch.float32),
    }
    output_tensor_configs = [
        {
            "name": "scores",
            "data_type": "TYPE_FP32",
            "dims": [-1, -1, 1],
            "label_filename": "",
            "is_shape_tensor": False,
        }
    ]

    response = build_inference_response(tensor_dict, output_tensor_configs)

    assert len(response.output_tensors) == 1
    pb_tensor = response.output_tensors[0]
    assert pb_tensor.name() == "scores"
    assert pb_tensor.as_numpy().dtype == np.float32
    np.testing.assert_array_equal(
        pb_tensor.as_numpy().shape, (batch_size, num_queries, 1)
    )
