# Dual-Camera RTSP Person ReID Pipeline

GPU-accelerated person re-identification across two CCTV cameras using
NVIDIA DeepStream 7.0, PeopleNet, NvDCF tracker, and a ResNet-50 ReID model.

---

## Architecture

```
RTSP Cam 0 ──┐
              ├─▶ nvstreammux ─▶ PeopleNet (pgie) ─▶ NvDCF tracker ─▶ ReID (sgie)
RTSP Cam 1 ──┘                                                              │
                                                                            ▼
                                                              pad probe (feature extraction)
                                                                            │
                                                              FeatureStore (per-camera)
                                                                            │
                                                          CrossCameraReIDMatcher (Hungarian)
                                                                            │
                                                              Global ID overlay (nvdsosd)
                                                                            │
                                                                    autovideosink
```

---

## Prerequisites

### 1. System packages

```bash
sudo apt-get install -y \
  libcairo2-dev \
  libgirepository-2.0-dev \
  libglib2.0-dev \
  pkg-config \
  python3-dev \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev
```

### 2. NVIDIA DeepStream SDK 7.0

Download the `.deb` from https://developer.nvidia.com/deepstream-sdk and install:

```bash
sudo apt-get install -y ./deepstream-7.0_7.0.0-1_amd64.deb
```

### 3. DeepStream Python bindings (`pyds`)

```bash
export PYTHONPATH=/opt/nvidia/deepstream/deepstream/lib:$PYTHONPATH
# Add the above line to your ~/.bashrc to make it permanent
python3 -c "import pyds; print('pyds OK')"
```

### 4. Python dependencies

```bash
uv sync
# or: pip install numpy scipy pygobject
```

### 5. Models from NVIDIA NGC

```bash
# Install NGC CLI: https://ngc.nvidia.com/setup/installers/cli
ngc config set   # enter your API key

# PeopleNet detector
ngc registry model download-version \
  "nvidia/tao/peoplenet:deployable_quantized_v2.6.1" \
  --dest ./models/peoplenet

# ReID model
ngc registry model download-version \
  "nvidia/tao/reidentificationnet:deployable_v1.0" \
  --dest ./models/reid
```

---

## Configuration

Edit `pipeline.py` and set your RTSP stream URIs:

```python
RTSP_SOURCES = [
    "rtsp://user:pass@192.168.1.10:554/stream1",   # Camera 0
    "rtsp://user:pass@192.168.1.11:554/stream1",   # Camera 1
]
```

Tune matching sensitivity in `pipeline.py`:

```python
matcher = CrossCameraReIDMatcher(
    similarity_threshold=0.65,   # lower = more permissive matching
    match_interval_sec=0.5       # how often cross-cam matching runs
)
```

---

## Running

```bash
export PYTHONPATH=/opt/nvidia/deepstream/deepstream/lib:$PYTHONPATH
python3 pipeline.py
```

---

## Project Structure

```
re-id/
├── pipeline.py          # Main GStreamer/DeepStream pipeline
├── probe_handler.py     # GStreamer pad probe: extracts ReID tensors
├── reid/
│   ├── feature_store.py # Thread-safe per-camera track feature store
│   ├── matcher.py       # Cross-camera Hungarian matching
│   └── gallery.py       # Long-term identity gallery (re-entry support)
├── config/
│   ├── peoplenet_infer.txt
│   ├── reid_infer.txt
│   └── tracker_NvDCF.yml
├── models/              # Downloaded from NGC (not in git)
├── pyproject.toml
└── README.md
```

---

## Key Design Notes

| Component | Detail |
|---|---|
| Probe placement | After `sgie` src pad — the only point where `NvDsInferTensorMeta` is populated |
| `output-tensor-meta=1` | Required in `reid_infer.txt` or probe sees no features |
| Camera ID source | `frame_meta.pad_index` from nvstreammux — not hardcoded |
| Feature normalisation | L2-normalised at extraction; mean feature re-normalised before matching |
| Matching throttle | Matcher runs at most every 500ms; returns cached assignments between runs |
| Stale track purge | Runs opportunistically inside `FeatureStore.update()` every 60s |
| Gallery | Persists embeddings after track loss; enables re-entry identification |