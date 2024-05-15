import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import tritonclient.grpc as grpcclient
import tritonclient.utils.cuda_shared_memory as cudashm
import tritonclient.utils.shared_memory as shm
import yaml
from loguru import logger
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor
from utils.client import (
    INPUT_WITH_CUDASHM_MODE,
    INPUT_WITH_SHM_MODE,
    INPUT_WITHOUT_SHM_MODE,
    MODEL_HOSTING_PATTERN_ALL_IN_ONE,
    MODEL_HOSTING_PATTERN_ENSEMBPLE_PY,
    MODEL_HOSTING_PATTERN_ENSEMBPLE_PY_ONNX_PY,
    MODEL_HOSTING_PATTERN_ENSEMBPLE_PY_PY_PY,
    MODEL_HOSTING_PATTERN_LOCAL,
    create_batch_tensor,
    create_input_tensors_with_cudashm,
    create_input_tensors_with_shm,
    create_input_tensors_wo_shared_memory,
    load_image,
    print_detected_objects,
    print_output_tensor_metas,
)

from triton_benchmarks.preprocss_inference_ensemble.detr.huggingface.wrapper.detection_service import (
    detr_end2end_inference,
    init_detr_model,
)
from triton_benchmarks.preprocss_inference_ensemble.triton.triton_utils import (
    convert_triton_stats_to_dataframe,
    summarize_triton_stats,
)

SAMPLE_IMAGE_PATH = (
    Path(__file__).parent / "samples" / "pexels-anna-tarazevich-14751175-fullhd.jpg"
)


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
        "--input-memory-type",
        type=str,
        default=INPUT_WITH_CUDASHM_MODE,
        help="Meomory type of the input tensor.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("./outputs/"),
        help="Path to a log directory.",
    )
    return parser


def prepare_input_tensor(
    input_image_path: Path,
    input_memory_type: str,
    batch_size: int,
    client: grpcclient.InferenceServerClient | None = None,
    host_pattern_no: int = MODEL_HOSTING_PATTERN_LOCAL,
):
    # Load image data.
    image_data = load_image(input_image_path)
    logger.info(f"Input Image: {image_data.shape} ({image_data.dtype})")
    batched_image = create_batch_tensor(image_data, batch_size=batch_size)
    logger.info(f"Input Tensor: {batched_image.shape} ({batched_image.dtype})")
    if host_pattern_no == MODEL_HOSTING_PATTERN_LOCAL:
        return batched_image, None

    # == Make InferInput ==
    if input_memory_type == INPUT_WITHOUT_SHM_MODE:
        input_tensors = create_input_tensors_wo_shared_memory(batched_image)
        shm_handles = None
    elif input_memory_type == INPUT_WITH_SHM_MODE:
        client.unregister_system_shared_memory()
        input_tensors, shm_handles = create_input_tensors_with_shm(
            batched_image, client
        )
    elif input_memory_type == INPUT_WITH_CUDASHM_MODE:
        client.unregister_system_shared_memory()
        client.unregister_cuda_shared_memory()
        input_tensors, shm_handles = create_input_tensors_with_cudashm(
            batched_image, client
        )
    else:
        raise ValueError(f"Invalid input_memory_type: {input_memory_type}")

    return input_tensors, shm_handles


def call_detr_model(
    host_pattern_no,
    input_tensors: torch.Tensor | list[grpcclient.InferInput],
    model: DetrForObjectDetection | None = None,
    processor: DetrImageProcessor | None = None,
    client: grpcclient.InferenceServerClient | None = None,
) -> tuple[dict[str, torch.Tensor] | list[grpcclient.InferRequestedOutput], float]:
    ts_start = time.time()
    if host_pattern_no == MODEL_HOSTING_PATTERN_LOCAL:
        results = detr_end2end_inference(model, processor, input_tensors)
    elif host_pattern_no == MODEL_HOSTING_PATTERN_ALL_IN_ONE:
        results = client.infer(model_name="all_in_one_detr", inputs=input_tensors)
    elif host_pattern_no == MODEL_HOSTING_PATTERN_ENSEMBPLE_PY:
        results = client.infer(model_name="pattern_1", inputs=input_tensors)
    elif host_pattern_no == MODEL_HOSTING_PATTERN_ENSEMBPLE_PY_PY_PY:
        results = client.infer(model_name="pattern_2", inputs=input_tensors)
    elif host_pattern_no == MODEL_HOSTING_PATTERN_ENSEMBPLE_PY_ONNX_PY:
        results = client.infer(model_name="pattern_3", inputs=input_tensors)
    else:
        raise ValueError(f"Invalid host_pattern_no: {host_pattern_no}")

    elapsed_time = time.time() - ts_start
    return results, elapsed_time


def main():
    args = make_parser().parse_args()
    logdir = (
        args.logdir
        / f"b{args.batch_size}-r{args.num_requests}"
        / f"p{args.model_hosting_pattern_no}"
    )
    logdir.mkdir(parents=True, exist_ok=True)

    if args.model_hosting_pattern_no == MODEL_HOSTING_PATTERN_LOCAL:
        model, processor = init_detr_model()
        client = None
    else:
        model, processor = None, None
        client = grpcclient.InferenceServerClient(url="localhost:8001")
    input_tensors, shm_handles = prepare_input_tensor(
        args.input,
        args.input_memory_type,
        args.batch_size,
        client,
        host_pattern_no=args.model_hosting_pattern_no,
    )

    logger.info(f"Call DETR model with pattern {args.model_hosting_pattern_no}")
    df_stats = []
    for _ in tqdm(range(args.num_requests)):
        results, elapsed_time = call_detr_model(
            args.model_hosting_pattern_no, input_tensors, model, processor, client
        )
        df_stats.append({"elapsed_time": elapsed_time})
    df_stats = pd.DataFrame(df_stats)
    df_stats["elapsed_time"] = df_stats["elapsed_time"] * 1e3
    elapsed_time = float(df_stats["elapsed_time"].mean())
    elapsed_time_std = float(df_stats["elapsed_time"].std())
    logger.info(
        f"Inference Request Completed! (Average Elapsed Time: {elapsed_time:.3f}ms [std: {elapsed_time_std:.3f}])"
    )

    # == Get Inference Statistics ==
    stats: dict = (
        client.get_inference_statistics(as_json=True)
        if args.model_hosting_pattern_no != MODEL_HOSTING_PATTERN_LOCAL
        else {}
    )
    stats["client"] = {
        "count": args.num_requests,
        "mean_msec": elapsed_time,
        "std_msec": elapsed_time_std,
        "mean_msec_wo_first_batch": float(df_stats["elapsed_time"].values[1:].mean()),
        "std_msec_wo_first_batch": float(df_stats["elapsed_time"].values[1:].std()),
        "min_msec": float(df_stats["elapsed_time"].min()),
        "max_msec": float(df_stats["elapsed_time"].max()),
        "elapsed_time": df_stats["elapsed_time"].tolist(),
    }
    yaml_path = logdir / "inference_statistics_raw.yaml"
    logger.info(f"Save stats to {yaml_path}")
    with open(yaml_path, "w") as f:
        yaml.dump(stats, f)

    if args.model_hosting_pattern_no != MODEL_HOSTING_PATTERN_LOCAL:
        df_stats: pd.DataFrame = convert_triton_stats_to_dataframe(stats)
        df_summary = summarize_triton_stats(df_stats)
        logger.info(f"Inference statistics summary:\n{df_summary.T}")

        csv_path = logdir / "inference_statistics_summary.csv"
        logger.info(f"Save stats to {csv_path}")
        df_summary.to_csv(csv_path)

    # == Postprocessing ==
    print_output_tensor_metas(results)
    print_detected_objects(results)

    # == Clean up ==
    if args.model_hosting_pattern_no != MODEL_HOSTING_PATTERN_LOCAL:
        client.unregister_system_shared_memory()
        client.unregister_cuda_shared_memory()
        if args.input_memory_type == INPUT_WITH_SHM_MODE:
            shm.destroy_shared_memory_region(shm_handles)
        elif args.input_memory_type == INPUT_WITH_CUDASHM_MODE:
            cudashm.destroy_shared_memory_region(shm_handles)
            assert len(cudashm.allocated_shared_memory_regions()) == 0


if __name__ == "__main__":
    main()
