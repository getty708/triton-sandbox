from pathlib import Path

import gi
import pyds
from gi.repository import GLib, Gst
from loguru import logger

gi.require_version("Gst", "1.0")


def generate_check_batch_size_probe(element_name: str):
    def check_batch_size_probe(pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        num_frames = batch_meta.num_frames_in_batch

        frame_infos = []
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_infos.append(
                f"Source{frame_meta.source_id}-Frame{frame_meta.frame_num}"
            )

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        logger.info(
            f"[{element_name:<10}] batch_size={num_frames}, frames={frame_infos}"
        )

        return Gst.PadProbeReturn.OK

    return check_batch_size_probe
