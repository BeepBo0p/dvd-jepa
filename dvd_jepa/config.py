"""Central configuration for DVD-JEPA.

Every constant lives here exactly once. All other modules import from here
instead of defining their own magic numbers.
"""

# ---------------------------------------------------------------------------
# World / simulation
# ---------------------------------------------------------------------------
FRAME_HEIGHT: int = 16
FRAME_WIDTH: int = 16
EMBEDDING_DIM: int = 32
BLOB_SIGMA: float = 1.5        # Gaussian blob radius in pixels

# Spawn margin: blob initial positions are drawn from [MARGIN, HEIGHT-MARGIN).
# Keeps the blob fully visible at the start of every episode.
SPAWN_MARGIN: float = 2.0

# Bounce boundary: velocity is flipped when the blob leaves [WALL_LOW, WALL_HIGH].
WALL_LOW: float = 1.0
WALL_HIGH: float = float(FRAME_HEIGHT - 2)   # 14.0; same for both axes since H == W

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
NUM_SEQUENCES: int = 400
SEQUENCE_LENGTH: int = 64
BLOB_VELOCITY: float = 0.05    # speed as a fraction of (frame_size - 1) per step
NOISE_STD_DEV: float = 0.05
TRAIN_FRACTION: float = 0.8

# ---------------------------------------------------------------------------
# Model hidden layer widths
# ---------------------------------------------------------------------------
ENCODER_HIDDEN_DIMS: tuple = (256, 128)   # 2*H*W -> 256 -> 128 -> EMBEDDING_DIM
PREDICTOR_HIDDEN_DIM: int = 64            # EMBEDDING_DIM -> 64 -> EMBEDDING_DIM
DECODER_HIDDEN_DIMS: tuple = (64, 256)    # EMBEDDING_DIM -> 64 -> 256 -> H*W
BASELINE_HIDDEN_DIM: int = 256            # MLPBaseline: 2*H*W -> 256 -> 256 -> H*W

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
JEPA_TRAIN_STEPS: int = 2500
DECODER_TRAIN_STEPS: int = 2000
BASELINE_TRAIN_STEPS: int = 2500
PROBE_TRAIN_STEPS: int = 800
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-3
PROBE_LEARNING_RATE: float = 1e-2
EMA_DECAY: float = 0.99          # target encoder momentum
VARIANCE_LOSS_WEIGHT: float = 1.0  # coefficient on the VICReg variance term

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
FORECAST_HORIZON: int = 30           # autoregressive steps to roll out
TELEPORT_AT: int = 24                # frame index where the anomaly is injected
ANOMALY_SEQUENCE_LENGTH: int = 44    # total frames in the anomaly evaluation sequence

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR: str = "checkpoints"
ASSETS_DIR: str = "assets"
