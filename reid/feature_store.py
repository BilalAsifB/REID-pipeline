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
    global_id: Optional[int] = None
    last_seen: float = field(default_factory=time.time)
    bbox_history: List[tuple] = field(default_factory=list)

    def add_feature(self, feat: np.ndarray, max_features: int = 10):
        self.features.append(feat / (np.linalg.norm(feat) + 1e-6))  # L2 normalize
        if len(self.features) > max_features:
            self.features.pop(0)
        self.last_seen = time.time()

    @property
    def mean_feature(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        return np.mean(self.features, axis=0)


class FeatureStore:
    def __init__(self, ttl_seconds: float = 300.0):
        self._store: Dict[int, Dict[int, TrackRecord]] = defaultdict(dict)
        # {camera_id: {track_id: TrackRecord}}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds

    def update(self, camera_id: int, track_id: int,
               feature: np.ndarray, bbox: tuple):
        with self._lock:
            if track_id not in self._store[camera_id]:
                self._store[camera_id][track_id] = TrackRecord(
                    track_id=track_id, camera_id=camera_id
                )
            record = self._store[camera_id][track_id]
            record.add_feature(feature)
            record.bbox_history.append(bbox)
        return self._store[camera_id][track_id]

    def get_record(self, camera_id: int, track_id: int) -> Optional[TrackRecord]:
        with self._lock:
            return self._store[camera_id].get(track_id)

    def all_records(self) -> List[TrackRecord]:
        with self._lock:
            return [r for cam in self._store.values() for r in cam.values()]

    def purge_stale(self):
        now = time.time()
        with self._lock:
            for cam_id in list(self._store.keys()):
                stale = [tid for tid, r in self._store[cam_id].items()
                         if now - r.last_seen > self._ttl]
                for tid in stale:
                    del self._store[cam_id][tid]