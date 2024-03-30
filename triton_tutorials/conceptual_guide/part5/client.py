import argparse
from pathlib import Path

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from loguru import logger


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triton client")
    parser.add_argument("-i", "--input", type=Path, help="Path to input image")
    return parser


def load_image_with_cv2(image_path: Path, quality: int = 50) -> np.ndarray:
    image_data = cv2.imread(str(image_path))
    logger.info(
        f"Original Input Image: {image_data.shape} ({image_data.dtype}) "
        f"=> Size: {image_data.ravel().shape} x UINT8"
    )

    # Encode image data to JPEG format.
    _, image_data_encoded = cv2.imencode(
        ".jpeg", image_data, (cv2.IMWRITE_JPEG_QUALITY, quality)
    )
    image_data_encoded = image_data_encoded.tobytes()

    # Create a numpy array from the encoded image data.
    image_data = np.frombuffer(image_data_encoded, dtype=np.uint8)
    image_data = np.expand_dims(image_data, axis=0)
    logger.info(f"Input Image: {image_data.shape} ({image_data.dtype})")
    return image_data


def load_image_with_numpy(image_path: Path) -> np.ndarray:
    image_data = np.fromfile(str(image_path), dtype="uint8")
    image_data = np.expand_dims(image_data, axis=0)
    return image_data


def main():
    args = make_parser().parse_args()
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Load image data.
    # image_data = load_image_with_cv2(args.input, quality=50)
    image_data = load_image_with_numpy(args.input)
    logger.info(f"Input Image: {image_data.shape} ({image_data.dtype})")

    # Call the triton server.
    input_tensors = [grpcclient.InferInput("input_image", image_data.shape, "UINT8")]
    input_tensors[0].set_data_from_numpy(image_data)
    results = client.infer(model_name="ensemble_model", inputs=input_tensors)
    output_data = results.as_numpy("recognized_text").astype(str)
    logger.info(f"Results: {output_data}")


if __name__ == "__main__":
    main()
