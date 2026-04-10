#!/usr/bin/env python3
"""
Dual-camera RTSP Person ReID Pipeline using NVIDIA DeepStream.
"""

import sys
import ctypes
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GLib

import pyds
from reid.feature_store import FeatureStore
from reid.matcher import CrossCameraReIDMatcher
from probe_handler import make_pad_probe_handler

# ─── Configuration ────────────────────────────────────────────────────────────

RTSP_SOURCES = [
    "rtsp://user:pass@192.168.1.10:554/stream1",   # Camera 0
    "rtsp://user:pass@192.168.1.11:554/stream1",   # Camera 1
]

OUTPUT_WIDTH  = 1920
OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000


def build_pipeline(rtsp_sources, feature_store, matcher):
    Gst.init(None)
    pipeline = Gst.Pipeline()

    # ── Stream Muxer ──────────────────────────────────────────────────────────
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("width", OUTPUT_WIDTH)
    streammux.set_property("height", OUTPUT_HEIGHT)
    streammux.set_property("batch-size", len(rtsp_sources))
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("live-source", 1)
    pipeline.add(streammux)

    # ── RTSP Sources ──────────────────────────────────────────────────────────
    for i, uri in enumerate(rtsp_sources):
        source_bin = make_source_bin(i, uri)
        pipeline.add(source_bin)

        pad_name = f"sink_{i}"
        sinkpad = streammux.get_request_pad(pad_name)
        srcpad  = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)

        # Attach per-source probe for camera_id tagging
        probe_cb = make_pad_probe_handler(i, feature_store, matcher)
        sinkpad.add_probe(Gst.PadProbeType.BUFFER, probe_cb, 0)

    # ── Primary Inference: PeopleNet Detector ─────────────────────────────────
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", "config/peoplenet_infer.txt")
    pipeline.add(pgie)

    # ── Object Tracker: NvDCF ─────────────────────────────────────────────────
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", "config/tracker_NvDCF.yml")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("gpu-id", 0)
    tracker.set_property("enable-past-frame", 1)
    tracker.set_property("enable-batch-process", 1)
    pipeline.add(tracker)

    # ── Secondary Inference: ReID Feature Extractor ───────────────────────────
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-reid")
    sgie.set_property("config-file-path", "config/reid_infer.txt")
    pipeline.add(sgie)

    # ── On-Screen Display ─────────────────────────────────────────────────────
    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    pipeline.add(nvanalytics)

    osd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    osd.set_property("process-mode", 0)
    osd.set_property("display-text", 1)
    pipeline.add(osd)

    # ── Tiler: show both cameras side-by-side ─────────────────────────────────
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", 1)
    tiler.set_property("columns", len(rtsp_sources))
    tiler.set_property("width", OUTPUT_WIDTH)
    tiler.set_property("height", OUTPUT_HEIGHT)
    pipeline.add(tiler)

    # ── Converter + Output Sink ───────────────────────────────────────────────
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "converter")
    pipeline.add(nvvidconv)

    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    )
    pipeline.add(capsfilter)

    # Encode and stream out (RTSP re-stream or file)
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    encoder.set_property("bitrate", 4000000)
    pipeline.add(encoder)

    sink = Gst.ElementFactory.make("rtspclientsink", "rtsp-sink")
    # Or use: filesink, fakesink, autovideosink for local display
    sink = Gst.ElementFactory.make("autovideosink", "display-sink")
    pipeline.add(sink)

    # ── Link Pipeline ─────────────────────────────────────────────────────────
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie)
    sgie.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(capsfilter)
    capsfilter.link(osd)
    osd.link(sink)

    return pipeline


def make_source_bin(index, uri):
    """Creates a uridecodebin source element for an RTSP URI."""
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)

    bin_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    nbin.add_pad(bin_pad)
    return nbin


def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    if gstname.find("video") != -1:
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            bin_ghost_pad.set_target(decoder_src_pad)
        else:
            print("ERROR: Requires NVMM memory features.")
            sys.exit(1)


def decodebin_child_added(child_proxy, obj, name, user_data):
    if name.find("decodebin") != -1:
        obj.connect("child-added", decodebin_child_added, user_data)
    if name.find("source") != -1:
        obj.set_property("drop-on-latency", True)


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"Warning: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}: {debug}")
        loop.quit()
    return True


def main():
    feature_store = FeatureStore(ttl_seconds=300)
    matcher = CrossCameraReIDMatcher(
        similarity_threshold=0.65,
        match_interval_sec=0.5
    )

    pipeline = build_pipeline(RTSP_SOURCES, feature_store, matcher)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("Starting ReID pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down pipeline...")
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()