"""
reid/gallery.py

Long-term gallery of known person embeddings.

When a tracked person leaves all cameras their mean feature vector is
archived here.  When a new track appears, it is matched against the
gallery first — enabling re-identification of people who re-enter the
scene after their track was purged from the FeatureStore.
"""

import numpy as np
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GalleryEntry:
    global_id: int
    feature: np.ndarray          # unit-normalised mean feature
    last_seen: float = field(default_factory=time.time)
    observation_count: int = 1


class Gallery:
    """
    Thread-safe long-term identity store.

    Workflow
    --------
    1. When a track is confirmed (≥3 features), call `query()` to check
       whether this person has been seen before.
    2. If a match is found (similarity ≥ threshold), reuse that global_id.
    3. When a track disappears, call `archive()` to save its embedding.
    """

    def __init__(self,
                 similarity_threshold: float = 0.70,
                 max_entries: int = 500,
                 ttl_seconds: float = 3600.0):
        self._threshold = similarity_threshold
        self._max       = max_entries
        self._ttl       = ttl_seconds
        self._lock      = threading.RLock()
        self._entries:  Dict[int, GalleryEntry] = {}   # global_id → entry

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, feature: np.ndarray) -> Optional[int]:
        """
        Return the global_id of the closest gallery entry above threshold,
        or None if no match is found.
        """
        with self._lock:
            if not self._entries:
                return None

            ids      = list(self._entries.keys())
            gallery  = np.stack([self._entries[i].feature for i in ids])  # (N, D)
            sims     = gallery @ feature                                   # (N,)

            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self._threshold:
                gid = ids[best_idx]
                self._entries[gid].last_seen = time.time()
                return gid
            return None

    # ── Archive ────────────────────────────────────────────────────────────────

    def archive(self, global_id: int, feature: np.ndarray) -> None:
        """
        Save or update a person's embedding in the gallery.
        Uses exponential moving average to keep the embedding fresh.
        """
        with self._lock:
            if global_id in self._entries:
                entry = self._entries[global_id]
                # EMA update: weight new observation lightly
                alpha = 0.1
                updated = (1 - alpha) * entry.feature + alpha * feature
                norm = np.linalg.norm(updated)
                entry.feature = updated / norm if norm > 1e-6 else updated
                entry.last_seen = time.time()
                entry.observation_count += 1
            else:
                if len(self._entries) >= self._max:
                    self._evict_oldest()
                self._entries[global_id] = GalleryEntry(
                    global_id=global_id,
                    feature=feature.copy()
                )

    # ── Maintenance ────────────────────────────────────────────────────────────

    def purge_stale(self) -> int:
        now = time.time()
        with self._lock:
            stale = [gid for gid, e in self._entries.items()
                     if now - e.last_seen > self._ttl]
            for gid in stale:
                del self._entries[gid]
            return len(stale)

    def _evict_oldest(self) -> None:
        """Remove the entry with the oldest last_seen timestamp."""
        oldest_gid = min(self._entries, key=lambda g: self._entries[g].last_seen)
        del self._entries[oldest_gid]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "gallery_size": len(self._entries),
                "total_observations": sum(
                    e.observation_count for e in self._entries.values()
                ),
            }