"""Central configuration for the EpiScore redundancy pipeline."""

from pathlib import Path

# --- Source dataset -------------------------------------------------------
# Egocentric-10K: 85 factories, 192,900 clips, 1080p/30fps H.265, Apache-2.0.
# Layout: factory_XXX/workers/worker_XXX/factoryXXX_workerXXX_partNN.tar
# Each tar holds paired <key>.mp4 / <key>.json samples.
REPO_ID = "builddotai/Egocentric-10K"
REPO_TYPE = "dataset"
SHARD_SUFFIX = ".tar"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"
EMB_DIR = OUT_DIR / "embeddings"

# --- Frame sampling -------------------------------------------------------
# One frame every N seconds of source video. Redundancy is a property of
# scene content, not of motion detail, so sparse sampling is sufficient and
# keeps 10k hours tractable.
FRAME_INTERVAL_SEC = 2.0

# Cap per clip so one long video cannot dominate a shard's embedding budget.
MAX_FRAMES_PER_CLIP = 240

# --- Embedding model ------------------------------------------------------
# ViT-B/32 is small enough for a 4GB card and its embedding space is the
# de-facto standard for near-duplicate work, which matters when we publish.
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512

# NOTE: fp32 on purpose. This GPU is Pascal (GP107), where fp16 arithmetic
# runs at 1/64 rate — half precision would make it slower, not faster.
# Embeddings are *stored* as fp16; only compute stays fp32.
BATCH_SIZE = 32
DEVICE_PREFERENCE = "cuda"

# --- Redundancy measurement ----------------------------------------------
# Reported as a curve across thresholds, never as one hardcoded number.
# A published redundancy figure that hinges on a magic constant is not
# defensible, and this analysis is meant to survive scrutiny.
SIM_THRESHOLDS = (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98)

# Threshold used for the headline figure and for pulling example pairs that
# a human can eyeball. Calibrated in notes/calibration.md, not guessed.
HEADLINE_THRESHOLD = 0.92
