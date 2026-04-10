import numpy as np
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


@dataclass
class TrackRecord:
    track_id: int
    camera_id: int
    features: List[np.ndarray] = field(default_factory=list)
    global_id: Optional[int]   = None
    last_seen: float            = field(default_factory=time.time)
    bbox_history: List[tuple]  = field(default_factory=list)

    MAX_FEATURES  = 10
    MAX_BBOX_HIST = 30

    def add_feature(self, feat: np.ndarray) -> None:
        # Features are expected to be L2-normalised by the caller
        self.features.append(feat)
        if len(self.features) > self.MAX_FEATURES:
            self.features.pop(0)
        self.last_seen = time.time()

    def add_bbox(self, bbox: tuple) -> None:
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > self.MAX_BBOX_HIST:
            self.bbox_history.pop(0)

    @property
    def mean_feature(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        mean = np.mean(self.features, axis=0)
        # Re-normalise the mean so it's a unit vector
        norm = np.linalg.norm(mean)
        return (mean / norm) if norm > 1e-6 else mean

    @property
    def is_mature(self) -> bool:
        """A track is considered reliable once it has accumulated ≥3 features."""
        return len(self.features) >= 3


class FeatureStore:
    """
    Thread-safe per-camera store for track feature vectors.

    Layout: {camera_id: {track_id: TrackRecord}}
    """

    def __init__(self, ttl_seconds: float = 300.0):
        self._store: Dict[int, Dict[int, TrackRecord]] = defaultdict(dict)
        self._lock        = threading.RLock()
        self._ttl         = ttl_seconds
        self._last_purge  = time.time()
        self._purge_every = 60.0   # run purge at most once per minute

    # ── Write ──────────────────────────────────────────────────────────────────

    def update(self, camera_id: int, track_id: int,
               feature: np.ndarray, bbox: tuple) -> "TrackRecord":
        with self._lock:
            cam_store = self._store[camera_id]
            if track_id not in cam_store:
                cam_store[track_id] = TrackRecord(
                    track_id=track_id, camera_id=camera_id
                )
            record = cam_store[track_id]
            record.add_feature(feature)
            record.add_bbox(bbox)

            # Opportunistic stale-track purge (avoids a separate timer thread)
            now = time.time()
            if now - self._last_purge > self._purge_every:
                self._purge_stale_locked(now)

        return record

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_record(self, camera_id: int,
                   track_id: int) -> Optional[TrackRecord]:
        with self._lock:
            return self._store[camera_id].get(track_id)

    def snapshot(self) -> List[TrackRecord]:
        """
        Return a *copy* of all current records so the caller can safely
        iterate without holding the lock.

        FIX: previous all_records() returned a live generator inside the
        lock scope, which was released before the caller finished iterating.
        """
        with self._lock:
            return [
                record
                for cam in self._store.values()
                for record in cam.values()
            ]

    def mature_snapshot(self) -> List[TrackRecord]:
        """Like snapshot() but only returns tracks with enough features."""
        with self._lock:
            return [
                record
                for cam in self._store.values()
                for record in cam.values()
                if record.is_mature
            ]

    # ── Maintenance ────────────────────────────────────────────────────────────

    def purge_stale(self) -> int:
        """Public entry point; returns number of tracks removed."""
        with self._lock:
            return self._purge_stale_locked(time.time())

    def _purge_stale_locked(self, now: float) -> int:
        """Must be called with self._lock held."""
        removed = 0
        for cam_id in list(self._store.keys()):
            stale_ids = [
                tid
                for tid, record in self._store[cam_id].items()
                if now - record.last_seen > self._ttl
            ]
            for tid in stale_ids:
                del self._store[cam_id][tid]
                removed += 1
        self._last_purge = now
        return removed

    def stats(self) -> Dict[int, int]:
        """Return {camera_id: track_count} for diagnostics."""
        with self._lock:
            return {cam_id: len(tracks)
                    for cam_id, tracks in self._store.items()}