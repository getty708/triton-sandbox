#!/usr/bin/python
"""Ref: https://triton-inference-server.github.io/pytriton/0.5.1/initialization/
"""
import logging
import time
from pathlib import Path

import click
import numpy as np
from model import SAMPLE_TOY_MODEL_NAME, infer_fn_sample_toy_model
from pytriton.client import ModelClient
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--logdir",
    type=str,
    default="none",
    help="path to the log file",
)
@click.option("--log-verbose-level", type=int, default=3)
def blocking_mode(logdir: str, log_verbose_level: int):
    if logdir.lower() == "none":
        triton_config = TritonConfig(log_verbose=log_verbose_level)
    else:
        triton_config = TritonConfig(
            log_file=Path(logdir, "triton-blocking.log"),
            log_verbose=log_verbose_level,
        )

    with Triton(config=triton_config) as triton:
        triton.bind(
            model_name=SAMPLE_TOY_MODEL_NAME,
            infer_func=infer_fn_sample_toy_model,
            inputs=[Tensor(dtype=np.float32, shape=(-1,))],
            outputs=[
                Tensor(dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128),
        )
        triton.serve()


@cli.command()
@click.option(
    "--logdir",
    type=str,
    default="none",
    help="path to the log file",
)
@click.option("--log-verbose-level", type=int, default=0)
def background_mode(logdir: str, log_verbose_level: int):
    if logdir.lower() == "none":
        triton_config = TritonConfig(log_verbose=log_verbose_level)
    else:
        triton_config = TritonConfig(
            log_file=Path(logdir, "triton-blocking.log"),
            log_verbose=log_verbose_level,
        )

    # Create a Triton instance and start server in background.
    triton = Triton(config=triton_config)
    triton.bind(
        model_name=SAMPLE_TOY_MODEL_NAME,
        infer_func=infer_fn_sample_toy_model,
        inputs=[Tensor(dtype=np.float32, shape=(-1,))],
        outputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
    )
    logger.info("=== Start Server ===")
    triton.run()
    time.sleep(5)

    # Inference
    logger.info("=== Inference ===")
    input1_batch = np.array([[1, 2]], dtype=np.float32)
    with ModelClient("localhost", SAMPLE_TOY_MODEL_NAME) as client:
        result_dict = client.infer_batch(input1_batch)
    for output_name, output_batch in result_dict.items():
        logger.info(f"{output_name}: {output_batch.tolist()}")
    np.testing.assert_array_equal(result_dict["OUTPUT_1"], [[2.0, 4.0]])

    # Stop the server
    logger.info("=== Stop Server ===")
    time.sleep(2)
    triton.stop()


if __name__ == "__main__":
    cli()
