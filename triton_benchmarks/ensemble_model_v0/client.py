import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import tritonclient.grpc as grpcclient
from loguru import logger
from tqdm import tqdm

from triton_benchmarks.ensemble_model_v0.utils.client import (
    cleanup_shared_memory,
    convert_results_on_cudashm_to_tensor_dict,
    create_input_output_tensors_with_cudashm,
    create_input_tensors_wo_shared_memory,
    create_logdir,
    prepare_image_batch_tensor,
    print_output_tensor_metas,
    save_client_stats,
    save_triton_inference_stats,
)
from triton_benchmarks.ensemble_model_v0.utils.const import ModelServePatern

_REPO_ROOT = Path(__file__).parents[2]
SAMPLE_IMAGE_PATH = _REPO_ROOT / "data" / "pexels-anna-tarazevich-14751175-fullhd.jpg"

MODEL_HOSTING_PATTERN_LOCAL = 100
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.remove()
logger.add(sys.stderr, level="INFO")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triton client")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=SAMPLE_IMAGE_PATH,
        help="Path to input image",
    )
    parser.add_argument(
        "-p",
        "--model-hosting-pattern-no",
        type=int,
        default=0,
        help="Hosting pattern of the DETR model.",
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=1,
        help="number of request send to the server.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=2,
        help="Batch size of the single request.",
    )
    parser.add_argument(
        "--use-shared-memory",
        action="store_true",
        help="Use shared memory or not.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("./outputs/"),
        help="Path to a log directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose mode.",
    )
    return parser


def prepare_input_tensor(
    image_batch: torch.Tensor,
    use_shared_memory: bool = True,
    client: grpcclient.InferenceServerClient | None = None,
):
    # == Make InferInput ==
    if use_shared_memory:
        if _DEVICE == "cpu":
            raise NotImplementedError
        else:
            client.unregister_system_shared_memory()
            client.unregister_cuda_shared_memory()
            input_tensors, output_tensors, shm_handles = (
                create_input_output_tensors_with_cudashm(image_batch, client)
            )
    else:
        input_tensors = create_input_tensors_wo_shared_memory(image_batch)
        output_tensors = None
        shm_handles = None

    return input_tensors, output_tensors, shm_handles


def call_dummy_model(
    host_pattern_no,
    input_tensors: torch.Tensor | list[grpcclient.InferInput],
    output_tensors: list[grpcclient.InferRequestedOutput] | None = None,
    client: grpcclient.InferenceServerClient | None = None,
) -> tuple[dict[str, torch.Tensor] | list[grpcclient.InferRequestedOutput], float]:
    ts_start = time.time()
    if host_pattern_no == ModelServePatern.NO_PIPELINE.value:
        results = client.infer(
            model_name="dummy_model", inputs=input_tensors, outputs=output_tensors
        )
    elif host_pattern_no in (
        ModelServePatern.PIPELINE_SIZE_1.value,
        ModelServePatern.PIPELINE_SIZE_2.value,
        ModelServePatern.PIPELINE_SIZE_3.value,
        ModelServePatern.PIPELINE_SIZE_4.value,
        ModelServePatern.PIPELINE_SIZE_5.value,
    ):
        results = client.infer(
            model_name=f"pipeline_{host_pattern_no}",
            inputs=input_tensors,
            outputs=output_tensors,
        )
    elif host_pattern_no in (
        ModelServePatern.SINGLE_MODEL_1.value,
        ModelServePatern.SINGLE_MODEL_2.value,
        ModelServePatern.SINGLE_MODEL_3.value,
        ModelServePatern.SINGLE_MODEL_4.value,
    ):
        results = client.infer(
            model_name=f"dummy_model_{host_pattern_no//10}",
            inputs=input_tensors,
            outputs=output_tensors,
        )
    else:
        raise ValueError(f"Invalid host_pattern_no: {host_pattern_no}")

    elapsed_time = time.time() - ts_start
    return results, elapsed_time


def main():
    args = make_parser().parse_args()
    logdir = create_logdir(args)

    # Prepare inputs
    image_batch = prepare_image_batch_tensor(args.input, args.batch_size)
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Inference
    logger.info(f"Call Triton with pattern {args.model_hosting_pattern_no}")
    df_stats = []
    for _ in tqdm(range(args.num_requests)):
        input_tensors, output_tensors, shm_handles = prepare_input_tensor(
            image_batch, use_shared_memory=args.use_shared_memory, client=client
        )
        results, elapsed_time = call_dummy_model(
            args.model_hosting_pattern_no, input_tensors, output_tensors, client
        )
        if args.verbose:
            output_tensors = convert_results_on_cudashm_to_tensor_dict(
                results, shm_handles[1]
            )
            print_output_tensor_metas(output_tensors)
        cleanup_shared_memory(client, shm_handles)
        df_stats.append({"elapsed_time": elapsed_time * 1e3})
    df_stats = pd.DataFrame(df_stats)
    logger.info(
        "Inference Request Completed! (Average Elapsed Time: {elapsed_time:.3f}ms [std: {elapsed_time_std:.3f}])".format(
            elapsed_time=df_stats["elapsed_time"].mean(),
            elapsed_time_std=df_stats["elapsed_time"].std(),
        )
    )

    # Save stats
    save_client_stats(logdir, df_stats)
    if args.model_hosting_pattern_no != MODEL_HOSTING_PATTERN_LOCAL:
        save_triton_inference_stats(logdir, client)


if __name__ == "__main__":
    main()
