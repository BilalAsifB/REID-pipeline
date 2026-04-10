import numpy as np
import threading
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from .feature_store import TrackRecord
import time


class CrossCameraReIDMatcher:
    """
    Matches person tracks across cameras using cosine similarity on
    aggregated (mean-pooled) ReID feature vectors.

    Matching runs at most once every `match_interval_sec` seconds to
    avoid blocking the GStreamer probe on every frame.  Between runs
    the most-recently computed assignment table is returned immediately.
    """

    def __init__(self,
                 similarity_threshold: float = 0.65,
                 match_interval_sec: float   = 0.5):
        self._threshold      = similarity_threshold
        self._match_interval = match_interval_sec
        self._lock           = threading.RLock()

        self._global_id_counter: int = 0

        # Stable assignment table: (camera_id, track_id) -> global_id
        # FIX: previously this was computed fresh but never persisted between
        # throttled calls, so the returned dict was empty on fast frames.
        self._assignments: Dict[Tuple[int, int], int] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def match(self, records: List[TrackRecord]) -> Dict[Tuple[int, int], int]:
        """
        Run (or skip) cross-camera matching and return the current
        (camera_id, track_id) -> global_id mapping.

        Only mature tracks (≥3 accumulated features) participate in
        cross-camera matching; immature tracks still get a local global_id.
        """
        now = time.time()

        with self._lock:
            # Always ensure every track has *some* global_id so the overlay
            # never shows -1 for long.
            self._ensure_all_assigned(records)

            # Throttle: skip expensive matching if called too frequently
            if now - getattr(self, "_last_match_time", 0.0) < self._match_interval:
                return dict(self._assignments)

            self._last_match_time = now
            self._run_cross_camera_match(records)
            return dict(self._assignments)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _next_global_id(self) -> int:
        self._global_id_counter += 1
        return self._global_id_counter

    def _ensure_all_assigned(self, records: List[TrackRecord]) -> None:
        """Give a global_id to any record that doesn't have one yet."""
        for r in records:
            key = (r.camera_id, r.track_id)
            if r.global_id is None:
                gid = self._next_global_id()
                r.global_id = gid
                self._assignments[key] = gid
            elif key not in self._assignments:
                # Record already has an id (e.g. carried over from a previous
                # run) but isn't in the assignment table yet — sync it.
                self._assignments[key] = r.global_id

    def _run_cross_camera_match(self, records: List[TrackRecord]) -> None:
        """
        Build a cost matrix between every pair of cameras and solve the
        linear assignment problem.  Updates self._assignments in place.
        """
        # Group records by camera; only use mature tracks for cross-cam matching
        cam_records: Dict[int, List[TrackRecord]] = {}
        for r in records:
            if r.mean_feature is not None and r.is_mature:
                cam_records.setdefault(r.camera_id, []).append(r)

        cameras = sorted(cam_records.keys())

        # Nothing to match if fewer than 2 cameras have mature tracks
        if len(cameras) < 2:
            return

        # For each ordered pair of cameras, run bipartite matching
        # (generalises cleanly to N cameras)
        for idx_a in range(len(cameras)):
            for idx_b in range(idx_a + 1, len(cameras)):
                cam_a = cameras[idx_a]
                cam_b = cameras[idx_b]
                self._match_camera_pair(
                    cam_records[cam_a],
                    cam_records[cam_b]
                )

    def _match_camera_pair(self,
                            records_a: List[TrackRecord],
                            records_b: List[TrackRecord]) -> None:
        """
        Solve the assignment problem between two sets of tracks.
        Only merge global_ids when cosine similarity ≥ self._threshold.
        """
        if not records_a or not records_b:
            return

        # Build cost matrix (1 - cosine_similarity)
        feats_a = np.stack([r.mean_feature for r in records_a])  # (Na, D)
        feats_b = np.stack([r.mean_feature for r in records_b])  # (Nb, D)

        # Efficient batched cosine similarity: since features are unit vectors,
        # dot product == cosine similarity
        sim_matrix  = feats_a @ feats_b.T          # (Na, Nb)
        cost_matrix = 1.0 - sim_matrix             # minimise → maximise similarity

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            similarity = sim_matrix[i, j]
            if similarity < self._threshold:
                continue   # Not the same person — leave IDs independent

            ra, rb = records_a[i], records_b[j]

            # Merge: prefer the lower (earlier-assigned) global_id
            if ra.global_id is not None and rb.global_id is not None:
                if ra.global_id == rb.global_id:
                    continue   # Already matched
                # Keep the smaller id (earlier assigned = more established)
                canonical = min(ra.global_id, rb.global_id)
                ra.global_id = canonical
                rb.global_id = canonical
            elif ra.global_id is not None:
                rb.global_id = ra.global_id
            elif rb.global_id is not None:
                ra.global_id = rb.global_id
            else:
                gid = self._next_global_id()
                ra.global_id = gid
                rb.global_id = gid

            self._assignments[(ra.camera_id, ra.track_id)] = ra.global_id
            self._assignments[(rb.camera_id, rb.track_id)] = rb.global_id

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def unique_person_count(self) -> int:
        """Return the number of distinct global IDs currently tracked."""
        with self._lock:
            return len(set(self._assignments.values()))

    def remove_track(self, camera_id: int, track_id: int) -> None:
        """Call this when a track disappears to keep the table clean."""
        with self._lock:
            self._assignments.pop((camera_id, track_id), None)