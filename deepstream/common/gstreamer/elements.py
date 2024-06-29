from typing import Callable

import gi
from gi.repository import Gst
from loguru import logger

gi.require_version("Gst", "1.0")


def create_gst_element(element_type: str, name: str) -> Gst.Element:
    element = Gst.ElementFactory.make(element_type, name)
    if not element:
        raise ValueError(f"Unable to create {element_type}")
    return element


def attach_sink_prob_element(gst_elem: Gst.Element, probe_func: Callable):
    sink_pad = gst_elem.get_static_pad("sink")
    sink_pad.add_probe(Gst.PadProbeType.BUFFER, probe_func, 0)
