from pathlib import Path

import click
import cv2
from loguru import logger

DATA_DIR = Path(__file__).parents[1]

SAMPLE_IMG_PATH = DATA_DIR / "pexels-anna-tarazevich-14751175-fullhd.jpg"


@click.command()
@click.option("--image-path", type=Path, default=SAMPLE_IMG_PATH)
@click.option(
    "--output-path", type=Path, default=DATA_DIR / f"{SAMPLE_IMG_PATH.stem}.mp4"
)
@click.option(
    "--duration", type=int, default=30, help="duration of the video [seconds]"
)
@click.option("--fps", type=int, default=1, help="frame rate of the video [seconds]")
def create_video(image_path: Path, output_path: Path, duration: int = 10, fps: int = 1):
    assert image_path.exists(), f"Image not found: {image_path}"

    # Load the image
    image = cv2.imread(image_path)

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_size = (image.shape[1], image.shape[0])  # width, height
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    # Repeat the image for 2 seconds
    num_frames = int(fps * duration)
    center = (int(frame_size[0] / 2), int(frame_size[1] / 2))
    scale = 1.0  # scale factor
    for i in range(num_frames):
        angle = (360.0 / num_frames) * i  # rotation angle [degree]
        trans = cv2.getRotationMatrix2D(center, angle, scale)
        image_rotated = cv2.warpAffine(image, trans, frame_size)
        video_writer.write(image_rotated)

    # Release the VideoWriter object
    video_writer.release()
    logger.info(f"The video was saved to {output_path}")


if __name__ == "__main__":
    create_video()
