import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from triton_benchmarks.preprocss_inference_ensemble.utils.analysis import (
    plot_e2e_inference_speed,
    plot_proc_inference_speed,
    summarize_e2e_inference_speed,
    summarize_process_inference_speed,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make summary and figures from the outputs."
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help="Path to a log directory.",
    )
    return parser


def main():
    args = make_parser().parse_args()
    logdir = args.logdir
    logdir.mkdir(parents=True, exist_ok=True)

    # E2E inference speed
    df_summary: pd.DataFrame = summarize_e2e_inference_speed(logdir)
    path = logdir / "e2e_inference_speed_summary.csv"
    logger.info(f"Save the summary CSV to {path}")
    df_summary.to_csv(path, index=True)

    fig_path = logdir / "e2e_inference_speed_summary.png"
    plot_e2e_inference_speed(df_summary, fig_path)

    # Process inference speed
    df_proc_summary = summarize_process_inference_speed(logdir)
    path = logdir / "proc_inference_speed_summary.csv"
    logger.info(f"Save the process summary CSV to {path}")
    df_proc_summary.to_csv(path, index=True)

    fig_path = logdir / "proc_inference_speed_summary.png"
    plot_proc_inference_speed(df_proc_summary, fig_path)


if __name__ == "__main__":
    main()
