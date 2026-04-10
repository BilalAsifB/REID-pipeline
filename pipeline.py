#!/usr/bin/env python3
"""
Dual-camera RTSP Person ReID Pipeline using NVIDIA DeepStream.
"""

import sys
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  

import pyds
from reid.feature_store import FeatureStore
from reid.matcher import CrossCameraReIDMatcher
from probe_handler import make_pad_probe_handler

# ─── Configuration ────────────────────────────────────────────────────────────

RTSP_SOURCES = [
    "rtsp://user:pass@192.168.1.10:554/stream1",
    "rtsp://user:pass@192.168.1.11:554/stream1",
]

OUTPUT_WIDTH  = 1920
OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000


def build_pipeline(rtsp_sources, feature_store, matcher):
    pipeline = Gst.Pipeline()

    # ── Stream Muxer ──────────────────────────────────────────────────────────
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    if not streammux:
        raise RuntimeError("Failed to create nvstreammux. Is DeepStream installed?")
    streammux.set_property("width", OUTPUT_WIDTH)
    streammux.set_property("height", OUTPUT_HEIGHT)
    streammux.set_property("batch-size", len(rtsp_sources))
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("live-source", 1)
    pipeline.add(streammux)

    # ── RTSP Sources ──────────────────────────────────────────────────────────
    for i, uri in enumerate(rtsp_sources):
        source_bin = make_source_bin(i, uri)
        if not source_bin:
            raise RuntimeError(f"Failed to create source bin for URI: {uri}")
        pipeline.add(source_bin)

        pad_name = f"sink_{i}"
        sinkpad = streammux.request_pad_simple(pad_name)
        if not sinkpad:
            raise RuntimeError(f"Failed to get muxer sink pad: {pad_name}")
        srcpad = source_bin.get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link source bin {i} to streammux")

    # ── Primary Inference: PeopleNet Detector ─────────────────────────────────
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        raise RuntimeError("Failed to create nvinfer (pgie)")
    pgie.set_property("config-file-path", "config/peoplenet_infer.txt")
    pipeline.add(pgie)

    # ── Object Tracker: NvDCF ─────────────────────────────────────────────────
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        raise RuntimeError("Failed to create nvtracker")
    tracker.set_property(
        "ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    )
    tracker.set_property("ll-config-file", "config/tracker_NvDCF.yml")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("gpu-id", 0)
    tracker.set_property("enable-past-frame", 1)
    tracker.set_property("enable-batch-process", 1)
    pipeline.add(tracker)

    # ── Secondary Inference: ReID Feature Extractor ───────────────────────────
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-reid")
    if not sgie:
        raise RuntimeError("Failed to create nvinfer (sgie)")
    sgie.set_property("config-file-path", "config/reid_infer.txt")
    pipeline.add(sgie)

    # ── Tiler: show both cameras side-by-side ─────────────────────────────────
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
    if not tiler:
        raise RuntimeError("Failed to create nvmultistreamtiler")
    tiler.set_property("rows", 1)
    tiler.set_property("columns", len(rtsp_sources))
    tiler.set_property("width", OUTPUT_WIDTH)
    tiler.set_property("height", OUTPUT_HEIGHT)
    pipeline.add(tiler)

    # ── On-Screen Display ─────────────────────────────────────────────────────
    osd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not osd:
        raise RuntimeError("Failed to create nvdsosd")
    osd.set_property("process-mode", 0)
    osd.set_property("display-text", 1)
    pipeline.add(osd)

    # ── Converter: NVMM → system memory for display ───────────────────────────
    # nvvideoconvert stays in NVMM; a second convert brings it to CPU memory
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "converter1")
    pipeline.add(nvvidconv1)

    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "converter2")
    pipeline.add(nvvidconv2)

    # Caps to force I420 in system (CPU) memory for autovideosink
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string("video/x-raw, format=I420") 
    )
    pipeline.add(capsfilter)

    # ── Display Sink ──────────────────────────────────────────────────────────
    sink = Gst.ElementFactory.make("autovideosink", "display-sink")
    if not sink:
        raise RuntimeError("Failed to create autovideosink")
    sink.set_property("sync", False)
    pipeline.add(sink)

    # ── Link Pipeline ─────────────────────────────────────────────────────────
    # streammux → pgie → tracker → sgie → tiler → nvvidconv1 → osd → nvvidconv2 → capsfilter → sink
    elements = [streammux, pgie, tracker, sgie, tiler, nvvidconv1, osd, nvvidconv2, capsfilter, sink]
    for i in range(len(elements) - 1):
        if not elements[i].link(elements[i + 1]):
            raise RuntimeError(
                f"Failed to link {elements[i].get_name()} → {elements[i+1].get_name()}"
            )

    # ── Attach ReID probe AFTER sgie so features are populated ────────────────
    # Probe on sgie src pad — this is where tensor output meta is available
    sgie_src_pad = sgie.get_static_pad("src")
    if not sgie_src_pad:
        raise RuntimeError("Failed to get sgie src pad")

    # Single probe handles all cameras; camera_id is read from frame_meta.pad_index
    sgie_src_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        make_pad_probe_handler(feature_store, matcher),
        0
    )

    return pipeline


def make_source_bin(index, uri):
    """Creates a uridecodebin source bin for an RTSP URI."""
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
    if not uri_decode_bin:
        return None
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)

    bin_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    nbin.add_pad(bin_pad)
    return nbin


def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        return
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    if "video" in gstname:
        if features and features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                print(f"ERROR: Failed to set ghost pad target for {gstname}")
                sys.exit(1)
        else:
            print("ERROR: Decoder did not produce NVMM memory. "
                  "Check that the NVIDIA GStreamer plugins are installed.")
            sys.exit(1)


def decodebin_child_added(child_proxy, obj, name, user_data):
    if "decodebin" in name:
        obj.connect("child-added", decodebin_child_added, user_data)
    if "source" in name:
        # Prevent pipeline stall on slow/lossy RTSP streams
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
    Gst.init(None)

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
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Failed to set pipeline to PLAYING state")
        sys.exit(1)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("Shutting down pipeline...")
        pipeline.set_state(Gst.State.NULL)
        feature_store.purge_stale()


if __name__ == "__main__":
    main()