import pyds
import numpy as np
from reid.feature_store import FeatureStore
from reid.matcher import CrossCameraReIDMatcher


def make_pad_probe_handler(camera_id: int,
                            feature_store: FeatureStore,
                            matcher: CrossCameraReIDMatcher):
    """
    Returns a GStreamer pad probe callback for a given camera source.
    Extracts object metadata + ReID feature vectors from DeepStream.
    """

    def pad_probe(pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                track_id = obj_meta.object_id
                rect = obj_meta.rect_params
                bbox = (rect.left, rect.top, rect.width, rect.height)

                # Extract ReID feature from secondary GIE (gie-unique-id=2)
                classifier_meta_list = obj_meta.classifier_meta_list
                while classifier_meta_list:
                    try:
                        clf_meta = pyds.NvDsClassifierMeta.cast(
                            classifier_meta_list.data
                        )
                    except StopIteration:
                        break

                    if clf_meta.unique_component_id == 2:  # ReID GIE
                        # Feature tensor is in label_info_list for feature output
                        label_list = clf_meta.label_info_list
                        while label_list:
                            try:
                                label_info = pyds.NvDsLabelInfo.cast(label_list.data)
                                # For feature-type outputs, result_class_id encodes dims
                                # Access raw tensor via user meta
                            except StopIteration:
                                break
                            label_list = label_list.next

                    classifier_meta_list = classifier_meta_list.next

                # --- Feature extraction via NvDsUserMeta (tensor output) ---
                user_meta_list = obj_meta.obj_user_meta_list
                while user_meta_list:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                        if user_meta.base_meta.meta_type == \
                                pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                            tensor_meta = pyds.NvDsInferTensorMeta.cast(
                                user_meta.user_meta_data
                            )
                            # Get feature vector from output layer
                            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                            ptr = ctypes.cast(
                                pyds.get_ptr(layer.buffer),
                                ctypes.POINTER(ctypes.c_float)
                            )
                            feature_dim = 512  # ResNet50 ReID output dim
                            feature = np.ctypeslib.as_array(ptr, shape=(feature_dim,)).copy()

                            # Update feature store
                            record = feature_store.update(
                                camera_id, track_id, feature, bbox
                            )

                            # Run cross-camera matching
                            all_records = feature_store.all_records()
                            assignments = matcher.match(all_records)
                            global_id = assignments.get((camera_id, track_id), -1)

                            # Annotate object meta with global ID
                            obj_meta.text_params.display_text = \
                                f"GID:{global_id} T:{track_id}"

                    except Exception:
                        pass
                    user_meta_list = user_meta_list.next

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return pyds.NvDsPadProbeReturn.OK

    return pad_probe