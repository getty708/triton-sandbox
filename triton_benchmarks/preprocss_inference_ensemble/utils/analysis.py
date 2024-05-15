from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from loguru import logger
from matplotlib.gridspec import GridSpec

CLIENT_KEY_NAME = "client"
# Exclude the first inference result because it is usually slow.
# MEAN_MSEC_KEY_NAME = "mean_msec"
# STD_MSEC_KEY_NAME = "std_msec"
MEAN_MSEC_KEY_NAME = "mean_msec_wo_first_batch"
STD_MSEC_KEY_NAME = "std_msec_wo_first_batch"
COUNT_KEY_NAME = "count"

PATTERN_ID_KEY_NAME = "pattern_id"
PATTERN_NAME_KEY_NAME = "pattern_name"
CLIENT_MEAN_MSEC_KEY_NAME = "client.mean_msec"
CLIENT_STD_MSEC_KEY_NAME = "client.std_msec"
CLIENT_COUNT_KEY_NAME = "client.count"
MODEL_KEY_NAME = "model"

AVG_SUCCSESS_TIME_KEY_NAME = "success.mean_ms"

DEFAULT_PATTERNS = {
    "p0": "Local",
    "p1": "Triton/py(all-in-one)",
    "p2": "Triton/py-py-py",
    "p3": "Triton/py-onnx-py",
}

TRITON_PROCESS_NAMES = (
    "queue.mean_ms",
    "compute_input.mean_ms",
    "compute_infer.mean_ms",
    "compute_output.mean_ms",
)

PROC_NAME_TO_HATCH_PATTERN = {
    "queue.mean_ms": "++",
    "compute_input.mean_ms": "..",
    "compute_infer.mean_ms": "/",
    "compute_output.mean_ms": "*",
}

TH_FOR_PLOT_VALUE = 5  # [ms]


def summarize_e2e_inference_speed(
    output_dir: Path, patterns: dict[str, str] | None = None
) -> pd.DataFrame:
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    # Load the YAML files
    df = []
    for pattern_key, pattern_name in patterns.items():
        path = output_dir / pattern_key / "inference_statistics_raw.yaml"
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            df.append(
                {
                    PATTERN_ID_KEY_NAME: pattern_key,
                    PATTERN_NAME_KEY_NAME: pattern_name,
                    CLIENT_MEAN_MSEC_KEY_NAME: data[CLIENT_KEY_NAME][
                        MEAN_MSEC_KEY_NAME
                    ],
                    CLIENT_STD_MSEC_KEY_NAME: data[CLIENT_KEY_NAME][STD_MSEC_KEY_NAME],
                    CLIENT_COUNT_KEY_NAME: data[CLIENT_KEY_NAME][COUNT_KEY_NAME],
                }
            )

    df = pd.DataFrame(df)
    df = df.set_index(PATTERN_ID_KEY_NAME)
    return df


def summarize_process_inference_speed(
    output_dir: Path, patterns: dict[str, str] | None = None
) -> pd.DataFrame:
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    # Load the CSV files
    df = []
    for pattern_key, pattern_name in patterns.items():
        if pattern_key == "p0":
            continue
        path = output_dir / pattern_key / "inference_statistics_summary.csv"
        if not path.exists():
            logger.warning(f"Skip {path} because it does not exist.")
            continue
        df_summary = pd.read_csv(path)
        df_summary[PATTERN_ID_KEY_NAME] = pattern_key
        df_summary[PATTERN_NAME_KEY_NAME] = pattern_name
        df.append(df_summary)
    df = pd.concat(df, axis=0, ignore_index=True)
    df[MODEL_KEY_NAME] = (
        df[MODEL_KEY_NAME]
        .apply(lambda x: "end-to-end" if x.startswith("pattern_") else x)
        .replace("all_in_one_detr", "end-to-end")
    )
    df = df.set_index([PATTERN_ID_KEY_NAME, PATTERN_NAME_KEY_NAME, MODEL_KEY_NAME])
    return df


def plot_e2e_inference_speed(df: pd.DataFrame, output_path: Path):
    sns.set_theme("notebook", "whitegrid")
    fig, ax0 = plt.subplots(figsize=(8, 6))

    xlabels = [pname.replace("/", "\n") for pname in df[PATTERN_NAME_KEY_NAME].values]
    x = np.arange(len(xlabels))
    y = df[CLIENT_MEAN_MSEC_KEY_NAME].values
    std = df[CLIENT_STD_MSEC_KEY_NAME].values

    for pattern_idx in range(len(x)):
        ax0.bar(
            x[pattern_idx],
            y[pattern_idx],
            yerr=std[pattern_idx],
            capsize=5,
            color=f"C{pattern_idx}",
            alpha=1.0,
        )
        ax0.text(
            x[pattern_idx],
            10,
            f"{y[pattern_idx]:.1f}",
            color="white",
            ha="center",
            va="bottom",
            fontsize="small",
            fontweight="bold",
        )

    ax0.set_xlabel("Pattern", fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(xlabels, rotation=0, ha="center", fontweight="bold")

    ax0.set_ylabel("Mean Inference Time [msec]", fontweight="bold")
    ax0.set_ylim([0, (y + std).max()])
    yticks = np.arange(0, (y + std).max(), 100)
    yticks_minor = np.arange(0, (y + std).max(), 20)
    ax0.set_yticks(yticks)
    ax0.set_yticks(yticks_minor, minor=True)

    ax0.grid(which="minor", linestyle=":")
    ax0.set_title("End-to-End Inference Speed", fontsize="x-large", fontweight="bold")
    fig.tight_layout()
    logger.info(f"Save the figure to {output_path}")
    fig.savefig(output_path)


def plot_proc_inference_speed(
    df: pd.DataFrame, output_path: Path, patterns: dict[str, str] | None = None
):
    if patterns is None:
        patterns = DEFAULT_PATTERNS
        patterns.pop("p0")

    sns.set_theme("notebook", "whitegrid")
    fig = plt.figure(figsize=(15, 10))
    gs_master = GridSpec(1, len(patterns))

    df_e2e = df.xs("end-to-end", level=MODEL_KEY_NAME)

    y_max = 0
    pattern_keys = sorted(list(patterns.keys()))
    for pattern_idx, pattern_key in enumerate(pattern_keys):
        pattern_name = patterns[pattern_key]
        if pattern_key == "p0":
            continue
        df_pattern = df.xs(pattern_key, level=PATTERN_ID_KEY_NAME)
        ax0 = fig.add_subplot(gs_master[pattern_idx])

        # Plot End-to-End Inference Time
        df_e2e = df_pattern.xs("end-to-end", level=MODEL_KEY_NAME)
        y_cumsum = 0
        for proc_idx, proc_name in enumerate(TRITON_PROCESS_NAMES):
            y = df_e2e[proc_name].values[0]
            ax0.bar(
                pattern_idx,
                y,
                label=f"E2E/{proc_name}",
                bottom=y_cumsum,
                color="C0",
                hatch=PROC_NAME_TO_HATCH_PATTERN[proc_name],
                align="edge",
                alpha=1.0,
                width=0.45,
            )
            if y >= TH_FOR_PLOT_VALUE:
                ax0.text(
                    pattern_idx + 0.225,
                    y_cumsum + y / 2,
                    f"{y:.1f}",
                    color="black",
                    ha="center",
                    va="center",
                    fontsize="small",
                    fontweight="bold",
                )
            y_cumsum += y
            if y_max < y_cumsum:
                y_max = y_cumsum

        # Plot Non-End-to-End Inference Time
        model_names = list(df_pattern.reset_index()[MODEL_KEY_NAME].unique())
        model_names.remove("end-to-end")
        y_cumsum = 0
        for model_idx, model_name in enumerate(model_names):
            df_model = df_pattern.xs(model_name, level=MODEL_KEY_NAME)
            for proc_idx, proc_name in enumerate(TRITON_PROCESS_NAMES):
                y = df_model[proc_name].values[0]
                ax0.bar(
                    pattern_idx + 0.50,
                    y,
                    label=f"{model_name}/{proc_name}",
                    bottom=y_cumsum,
                    color=f"C{model_idx+1}",
                    hatch=PROC_NAME_TO_HATCH_PATTERN[proc_name],
                    align="edge",
                    alpha=1.0,
                    width=0.45,
                )
                if y >= TH_FOR_PLOT_VALUE:
                    ax0.text(
                        pattern_idx + 0.50 + 0.225,
                        y_cumsum + y / 2,
                        f"{y:.1f}",
                        color="black",
                        ha="center",
                        va="center",
                        fontsize="small",
                        fontweight="bold",
                    )
                y_cumsum += y
                if y_max < y_cumsum:
                    y_max = y_cumsum

        ax0.set_ylabel("Avg. Time [ms]", fontweight="bold")
        ax0.set_ylim([0, y_max])
        yticks = np.arange(0, y_max, 50)
        yticks_minor = np.arange(0, y_max, 50)
        ax0.set_yticks(yticks)
        ax0.set_yticks(yticks_minor, minor=True)

        ax0.legend(
            loc="upper center",
            fontsize="small",
            bbox_to_anchor=(0.5, -0.01),
            ncol=1,
        )
        ax0.grid(which="minor", linestyle=":")
        ax0.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax0.set_title(
            f"({pattern_key}) {pattern_name}", fontsize="x-large", fontweight="bold"
        )

    fig.tight_layout()
    logger.info(f"Save the figure to {output_path}")
    fig.savefig(output_path)
