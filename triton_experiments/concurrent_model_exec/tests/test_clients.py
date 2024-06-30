# pylint: disable=C0415
import pytest
import torch
from click.testing import CliRunner
from pytest_mock import MockFixture


@pytest.mark.parametrize("model_name,num_output_tensors", [("single", 1), ("ensemble_", 3)])
def test_prepare_input_tensor(
    mocker: MockFixture,
    model_name: str,
    num_output_tensors: int,
):
    # When tritonclient.utils.shared_memory is imported, C libiraries are imported but not available in the test environment.
    # Import module with patching triton_experiments.common.triton_client.utils.
    mock_utils = mocker.MagicMock()
    mocker.patch.dict("sys.modules", {"triton_experiments.common.triton_client.utils": mock_utils})
    from triton_experiments.concurrent_model_exec.client import prepare_input_tensor

    batched_image = torch.rand((1, 3, 224, 224))
    mock_client = mocker.patch("tritonclient.grpc.InferenceServerClient")
    mocker.patch("tritonclient.grpc.InferenceServerClient.unregister_system_shared_memory")
    mocker.patch("tritonclient.grpc.InferenceServerClient.unregister_cuda_shared_memory")
    mocked_create_input_output_tensors = mocker.patch(
        "triton_experiments.concurrent_model_exec.client.create_input_output_tensors_with_cudashm",
        return_value=(None, None, None),
    )

    outputs = prepare_input_tensor(model_name, batched_image, mock_client)

    assert len(outputs) == 3
    _, kwargs = mocked_create_input_output_tensors.call_args
    assert len(kwargs["output_tensor_names"]) == num_output_tensors


def test_call_triton_model(mocker: MockFixture):
    mock_utils = mocker.MagicMock()
    mocker.patch.dict("sys.modules", {"triton_experiments.common.triton_client.utils": mock_utils})
    from triton_experiments.concurrent_model_exec.client import call_triton_model

    pipeline_name = "ensemble_sequential"
    input_tensors = [mocker.patch("tritonclient.grpc.InferInput")]
    output_tensors = [mocker.MagicMock()]
    mock_client = mocker.patch("tritonclient.grpc.InferenceServerClient")

    _, elapsed_time = call_triton_model(pipeline_name, input_tensors, output_tensors, mock_client)

    assert elapsed_time >= 0


def test_smoke_main(mocker: MockFixture):
    mock_utils = mocker.MagicMock()
    mocker.patch.dict("sys.modules", {"triton_experiments.common.triton_client.utils": mock_utils})
    from triton_experiments.concurrent_model_exec.client import main as target_main_func

    mocker.patch("tritonclient.grpc.InferenceServerClient", return_value=mocker.MagicMock())
    mocker.patch(
        "triton_experiments.concurrent_model_exec.client.prepare_input_tensor",
        return_value=(None, None, None),
    )
    mocker.patch(
        "triton_experiments.concurrent_model_exec.client.call_triton_model",
        return_value=(None, 0.0),
    )
    mocker.patch("triton_experiments.concurrent_model_exec.client.cleanup_shared_memory")

    runner = CliRunner()
    result = runner.invoke(target_main_func)

    assert result.exit_code == 0
