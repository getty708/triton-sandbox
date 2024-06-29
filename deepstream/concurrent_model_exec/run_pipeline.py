import sys
from pathlib import Path

import click
import gi
import pyds

from deepstream.common.gstreamer.elements import (
    attach_sink_prob_element,
    create_gst_element,
)
from deepstream.common.gstreamer.input_stream import create_source_bin
from deepstream.common.gstreamer.probe_funcs import generate_check_batch_size_probe

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
from loguru import logger

from deepstream.common.nvidia.FPS import PERF_DATA

MUXER_BATCH_TIMEOUT_MSEC = 10
CONFIG_DIR = Path(__file__).parent.resolve() / "configs"
SAMPLE_VIDEO_PATH = (
    "/workspace/triton-sandbox/data/pexels-anna-tarazevich-14751175-fullhd.mp4"
)
SAMPLE_VIDEO_URI = f"file://{SAMPLE_VIDEO_PATH}"

perf_data = None
measure_latency = False


def perf_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


@click.command()
@click.option("-i", "--input-video", type=str, default=SAMPLE_VIDEO_URI)
@click.option("-b", "--batch-size", type=int, default=1)
def main(
    input_video: str = SAMPLE_VIDEO_URI,
    batch_size: int = 8,
    model1_config_path: Path = Path("./configs/simple_cnn_l1.txt"),
    model2_config_path: Path = Path("./configs/simple_cnn_l2.txt"),
    model3_config_path: Path = Path("./configs/simple_cnn_l3.txt"),
):
    uri_list = [input_video] * batch_size
    # uri_list = [input_video.replace(".mp4", f"-s{i}.mp4") for i in range(batch_size)]
    logger.info(f"Input videos: {uri_list}")

    global perf_data
    perf_data = PERF_DATA(len(uri_list))

    # Standard GStreamer initialization
    Gst.init(None)

    # == Create Gstreamer Elements ==
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    logger.info(f"Create SrcBin and StreamMux")
    streammux = create_gst_element("nvstreammux", "stream-mux")
    streammux.set_property("batch-size", batch_size)
    streammux.set_property("sync-inputs", 0)
    streammux.set_property("max-latency", int(MUXER_BATCH_TIMEOUT_MSEC * 1e6))
    streammux_config_path = CONFIG_DIR / "streammux.txt"
    streammux.set_property("config-file-path", str(streammux_config_path))
    streammux.set_property("drop-pipeline-eos", 0)
    pipeline.add(streammux)
    for i in range(batch_size):
        logger.info(f"Creating source_bin {i}: {input_video}")
        source_bin = create_source_bin(i, input_video)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    # streammux = create_gst_element("nvmultiurisrcbin", "stream-mux")
    # streammux.set_property("port", 9000)
    # streammux.set_property("ip-address", "localhost")
    # streammux.set_property("batched-push-timeout", 100000)
    # streammux.set_property("max-batch-size", 8)
    # streammux.set_property("drop-pipeline-eos", 0)
    # streammux.set_property("live-source", 0)
    # streammux.set_property("mode", 0)
    # streammux.set_property("frame-duration", 1000)
    # # streammux.set_property("sync-inputs", 1) # This is not working
    # _uri_list = ",".join([str(input_video) for _ in range(batch_size)])
    # # _uri_list = str(input_video)
    # print(_uri_list)
    # streammux.set_property("uri-list", _uri_list)
    # streammux.set_property("width", 1920)
    # streammux.set_property("height", 1080)
    # pipeline.add(streammux)

    # Models
    model_1_queue = create_gst_element("queue", "model-1-queue")
    model_1_queue.set_property("max-size-buffers", 1)
    attach_sink_prob_element(
        model_1_queue, generate_check_batch_size_probe("model-1-queue")
    )
    pipeline.add(model_1_queue)

    model_1 = create_gst_element("nvinfer", "model-1")
    model_1.set_property("config-file-path", str(model1_config_path))
    model_1.set_property("unique-id", 1)
    model_1.set_property("process-mode", 1)
    model_1.set_property("output-tensor-meta", 1)
    model_1.set_property("batch-size", batch_size)
    attach_sink_prob_element(model_1, generate_check_batch_size_probe("model-1"))
    pipeline.add(model_1)

    model_2_queue = create_gst_element("queue", "model-2-queue")
    model_2_queue.set_property("max-size-buffers", 1)
    attach_sink_prob_element(
        model_2_queue, generate_check_batch_size_probe("model-2-queue")
    )
    pipeline.add(model_2_queue)

    model_2 = create_gst_element("nvinfer", "model-2")
    model_2.set_property("config-file-path", str(model2_config_path))
    model_2.set_property("unique-id", 2)
    model_2.set_property("process-mode", 1)
    model_2.set_property("output-tensor-meta", 1)
    model_2.set_property("batch-size", batch_size)
    pipeline.add(model_2)

    model_3_queue = create_gst_element("queue", "model-3-queue")
    model_3_queue.set_property("max-size-buffers", 1)
    attach_sink_prob_element(
        model_3_queue, generate_check_batch_size_probe("model-3-queue")
    )
    pipeline.add(model_3_queue)

    model_3 = create_gst_element("nvinfer", "model-3")
    model_3.set_property("config-file-path", str(model3_config_path))
    model_3.set_property("unique-id", 3)
    model_3.set_property("process-mode", 1)
    model_3.set_property("output-tensor-meta", 1)
    model_3.set_property("batch-size", batch_size)
    pipeline.add(model_3)

    # Fakesink
    logger.info(f"create sink element")
    sink_queue = create_gst_element("queue", "sink-queue")
    attach_sink_prob_element(sink_queue, generate_check_batch_size_probe("sink-queue"))
    pipeline.add(sink_queue)
    sink = create_gst_element("fakesink", "sink")
    # sink.set_property("dump", True)
    attach_sink_prob_element(sink, generate_check_batch_size_probe("sink"))
    pipeline.add(sink)

    # == Link ==
    logger.info(f"Link elements")
    streammux.link(model_1_queue)
    model_1_queue.link(model_1)
    # model_1.link(model_2)
    # model_2.link(model_3)
    model_1.link(model_2_queue)
    model_2_queue.link(model_2)
    model_2.link(model_3_queue)
    model_3_queue.link(model_3)
    model_3.link(sink_queue)
    sink_queue.link(sink)

    # == probe ==
    logger.info(f"Create perfrormance buffer probe to streammux src pad.")
    streammux_src_pad = streammux.get_static_pad("src")
    streammux_src_pad.add_probe(Gst.PadProbeType.BUFFER, perf_src_pad_buffer_probe, 0)
    GLib.timeout_add(5000, perf_data.perf_print_callback)

    # create and event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, "pipeline")

    # List the sources
    logger.info("Now playing...")
    for i, source in enumerate(uri_list):
        logger.info(f"- {i}: {source}")

    logger.info("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except Exception as e:
        logger.warning(f"Error: {e}")
        pass

    logger.info("EOS. Clean up the pipeline.\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
