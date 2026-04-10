import numpy as np
import threading
from typing import Dict, Optional, List, Tuple
from scipy.optimize import linear_sum_assignment
from .feature_store import TrackRecord
import time


class CrossCameraReIDMatcher:
    """
    Matches person tracks across cameras using cosine similarity
    on aggregated ReID feature vectors.
    """

    def __init__(self,
                 similarity_threshold: float = 0.65,
                 match_interval_sec: float = 0.5):
        self._global_id_counter = 0
        self._lock = threading.RLock()
        self._similarity_threshold = similarity_threshold
        self._match_interval = match_interval_sec
        self._last_match_time = 0.0

        # global_id -> list of (camera_id, track_id)
        self._global_registry: Dict[int, List[Tuple[int, int]]] = {}

    def _next_global_id(self) -> int:
        self._global_id_counter += 1
        return self._global_id_counter

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def match(self, records: List[TrackRecord]) -> Dict[Tuple[int, int], int]:
        """
        Run cross-camera matching. Returns mapping of
        (camera_id, track_id) -> global_id
        """
        now = time.time()
        if now - self._last_match_time < self._match_interval:
            # Return existing assignments without re-running
            return self._current_assignments(records)

        self._last_match_time = now

        with self._lock:
            # Split by camera
            cam_records: Dict[int, List[TrackRecord]] = {}
            for r in records:
                cam_records.setdefault(r.camera_id, []).append(r)

            cameras = list(cam_records.keys())
            if len(cameras) < 2:
                # Single camera — just assign global IDs locally
                for r in records:
                    if r.global_id is None:
                        r.global_id = self._next_global_id()
                return self._current_assignments(records)

            # Build cross-camera cost matrix for camera pair (0, 1)
            cam_a_records = [r for r in cam_records[cameras[0]] if r.mean_feature is not None]
            cam_b_records = [r for r in cam_records[cameras[1]] if r.mean_feature is not None]

            if cam_a_records and cam_b_records:
                cost = np.zeros((len(cam_a_records), len(cam_b_records)))
                for i, ra in enumerate(cam_a_records):
                    for j, rb in enumerate(cam_b_records):
                        cost[i, j] = 1.0 - self.cosine_similarity(
                            ra.mean_feature, rb.mean_feature
                        )

                row_ind, col_ind = linear_sum_assignment(cost)

                for i, j in zip(row_ind, col_ind):
                    similarity = 1.0 - cost[i, j]
                    ra, rb = cam_a_records[i], cam_b_records[j]

                    if similarity >= self._similarity_threshold:
                        # Same person across cameras
                        if ra.global_id is not None:
                            rb.global_id = ra.global_id
                        elif rb.global_id is not None:
                            ra.global_id = rb.global_id
                        else:
                            gid = self._next_global_id()
                            ra.global_id = gid
                            rb.global_id = gid

            # Assign unmatched records their own global IDs
            for r in records:
                if r.global_id is None:
                    r.global_id = self._next_global_id()

        return self._current_assignments(records)

    def _current_assignments(self, records: List[TrackRecord]) -> Dict[Tuple[int, int], int]:
        return {
            (r.camera_id, r.track_id): r.global_id
            for r in records if r.global_id is not None
        }