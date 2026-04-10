import ctypes  # FIX: was missing — needed for pointer casting
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pyds
from reid.feature_store import FeatureStore
from reid.matcher import CrossCameraReIDMatcher

# RGBA colour for the OSD text overlay (green, fully opaque)
_TEXT_COLOR = {"red": 0.0, "green": 1.0, "blue": 0.0, "alpha": 1.0}
_FONT_SIZE  = 12


def _set_display_text(obj_meta: pyds.NvDsObjectMeta, text: str) -> None:
    """
    Safely write display text into an object's text_params.
    Must be called while the GIL is held (normal Python execution).
    """
    txt = obj_meta.text_params
    txt.display_text    = text
    txt.font_params.font_name = "Serif"
    txt.font_params.font_size = _FONT_SIZE
    txt.font_params.font_color.set(
        _TEXT_COLOR["red"],
        _TEXT_COLOR["green"],
        _TEXT_COLOR["blue"],
        _TEXT_COLOR["alpha"],
    )
    txt.set_bg_clr      = 1
    txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)   # semi-transparent black background
    txt.x_offset        = int(obj_meta.rect_params.left)
    txt.y_offset        = max(0, int(obj_meta.rect_params.top) - _FONT_SIZE - 4)


def make_pad_probe_handler(feature_store: FeatureStore,
                            matcher: CrossCameraReIDMatcher):
    """
    Returns a single GStreamer pad probe callback to be attached to the
    sgie (ReID secondary inference) src pad.

    IMPORTANT: This probe must be placed AFTER sgie — not on the muxer
    sink pads — so that NvDsInferTensorMeta is already populated.

    Camera identity is derived from frame_meta.pad_index (set by
    nvstreammux), which maps 1:1 to the RTSP source index.
    """

    REID_GIE_UID   = 2      # must match gie-unique-id in reid_infer.txt
    FEATURE_DIM    = 512    # ResNet50 ReID output dimensionality

    def pad_probe(pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            # pad_index is set by nvstreammux and equals the RTSP source index
            camera_id = frame_meta.pad_index

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                track_id = obj_meta.object_id
                rect     = obj_meta.rect_params
                bbox     = (rect.left, rect.top, rect.width, rect.height)

                # ── Extract ReID feature tensor ────────────────────────────
                feature = _extract_reid_feature(
                    obj_meta, REID_GIE_UID, FEATURE_DIM
                )

                if feature is not None:
                    # Update per-camera feature store
                    feature_store.update(camera_id, track_id, feature, bbox)

                    # Run cross-camera matching (throttled internally)
                    all_records  = feature_store.snapshot()
                    assignments  = matcher.match(all_records)
                    global_id    = assignments.get((camera_id, track_id), -1)

                    # Annotate overlay
                    _set_display_text(
                        obj_meta,
                        f"GID:{global_id}  CAM:{camera_id}  TID:{track_id}"
                    )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK   # FIX: was pyds.NvDsPadProbeReturn.OK (doesn't exist)

    return pad_probe


def _extract_reid_feature(obj_meta: pyds.NvDsObjectMeta,
                           reid_gie_uid: int,
                           feature_dim: int):
    """
    Walk the object's user-meta list looking for NvDsInferTensorMeta
    produced by the ReID secondary GIE.

    Returns a normalised float32 numpy array of shape (feature_dim,),
    or None if the tensor is not yet available for this object.
    """
    user_meta_list = obj_meta.obj_user_meta_list
    while user_meta_list is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
        except StopIteration:
            break

        if user_meta.base_meta.meta_type == \
                pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            try:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(
                    user_meta.user_meta_data
                )

                # Verify this tensor comes from our ReID GIE
                if tensor_meta.unique_id != reid_gie_uid:
                    user_meta_list = user_meta_list.next
                    continue

                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                ptr   = ctypes.cast(
                    pyds.get_ptr(layer.buffer),
                    ctypes.POINTER(ctypes.c_float)
                )
                feature = np.ctypeslib.as_array(ptr, shape=(feature_dim,)).copy()

                # L2-normalise here so cosine similarity == dot product
                norm = np.linalg.norm(feature)
                if norm > 1e-6:
                    feature /= norm

                return feature.astype(np.float32)

            except Exception as e:
                print(f"[probe] Tensor extraction error: {e}")

        try:
            user_meta_list = user_meta_list.next
        except StopIteration:
            break

    return None