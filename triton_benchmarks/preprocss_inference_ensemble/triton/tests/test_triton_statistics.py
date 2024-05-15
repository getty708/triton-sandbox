import pandas as pd
import pytest

from triton_benchmarks.preprocss_inference_ensemble.triton.triton_utils import (
    convert_triton_stats_to_dataframe,
    summarize_triton_stats,
)


@pytest.fixture
def triton_stats_dict():
    stats = {
        "model_stats": [
            {
                "name": "detr_onnx",
                "version": "1",
                "last_inference": "1714524098299",
                "inference_count": "10",
                "execution_count": "10",
                "inference_stats": {
                    "success": {"count": "10", "ns": "500000000"},
                    "fail": {},
                    "queue": {"count": "10", "ns": "100000000"},
                    "compute_input": {"count": "10", "ns": "200000000"},
                    "compute_infer": {"count": "10", "ns": "300000000"},
                    "compute_output": {"count": "10", "ns": "400000000"},
                    "cache_hit": {},
                    "cache_miss": {},
                },
                "batch_stats": [
                    {
                        "batch_size": "1",
                        "compute_input": {"count": "10", "ns": "23997290"},
                        "compute_infer": {"count": "10", "ns": "54026184901"},
                        "compute_output": {"count": "10", "ns": "13995624"},
                    }
                ],
            },
            {
                "name": "detr_preprocessing",
                "version": "1",
                "last_inference": "1714524096071",
                "inference_count": "10",
                "execution_count": "10",
                "inference_stats": {
                    "success": {"count": "10", "ns": "5000000000"},
                    "fail": {},
                    "queue": {"count": "10", "ns": "1000000000"},
                    "compute_input": {"count": "10", "ns": "2000000000"},
                    "compute_infer": {"count": "10", "ns": "3000000000"},
                    "compute_output": {"count": "10", "ns": "4000000000"},
                    "cache_hit": {},
                    "cache_miss": {},
                },
                "batch_stats": [
                    {
                        "batch_size": "1",
                        "compute_input": {"count": "10", "ns": "334500083"},
                        "compute_infer": {"count": "10", "ns": "10686994005"},
                        "compute_output": {"count": "10", "ns": "657775377"},
                    }
                ],
            },
        ]
    }
    return stats


def test_convert_triton_stats_to_dataframe(triton_stats_dict: dict):
    df_actual = convert_triton_stats_to_dataframe(triton_stats_dict)

    print(df_actual.T)
    assert isinstance(df_actual, pd.DataFrame)
    assert df_actual.shape == (2, 14)
    assert all([col_dtype == int for col_dtype in df_actual.dtypes])


def test_summarize_triton_stats_dataframe(triton_stats_dict: dict):
    df_summary_expected = pd.DataFrame(
        {
            "success.mean_ms": [50.0, 500.0],
            "queue.mean_ms": [10.0, 100.0],
            "compute_input.mean_ms": [20.0, 200.0],
            "compute_infer.mean_ms": [30.0, 300.0],
            "compute_output.mean_ms": [40.0, 400.0],
        },
        index=["detr_onnx", "detr_preprocessing"],
    )
    df_summary_expected.index.name = "model"

    df_stats = convert_triton_stats_to_dataframe(triton_stats_dict)
    df_summary = summarize_triton_stats(df_stats)

    print(df_summary.T)

    pd.testing.assert_frame_equal(df_summary, df_summary_expected)
