from pathlib import Path

import pytest

from triton_benchmarks.preprocss_inference_ensemble.utils.analysis import (
    plot_e2e_inference_speed,
    summarize_e2e_inference_speed,
)


@pytest.fixture
def output_dir() -> Path:
    # TODO: Replace with dummy data
    return Path(__file__).parents[2] / "outputs" / "run1-b1"


def test_summarize_e2e_inference_speed(output_dir: Path):
    cols_expect = [
        "pattern_name",
        "client.mean_msec",
        "client.std_msec",
        "client.count",
    ]
    index_expect = ["p0", "p1", "p2", "p3"]

    df = summarize_e2e_inference_speed(output_dir)

    assert set(df.index) == set(index_expect)
    assert set(df.columns) == set(cols_expect)


def test_plot_e2e_inference_speed(output_dir: Path):
    df = summarize_e2e_inference_speed(output_dir)
    output_path = output_dir / "e2e_inference_speed_summary.png"

    plot_e2e_inference_speed(df, output_path)

    assert output_path.exists()
