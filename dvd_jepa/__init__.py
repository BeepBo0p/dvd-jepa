"""DVD-JEPA: a minimal, fully-reproducible Joint-Embedding Predictive
Architecture world model, trained on a bouncing DVD logo.

The package is intentionally tiny and dependency-light so the whole idea fits
in your head and trains on a CPU in under a minute. See README.md for details.
"""
from .config import (
    FRAME_HEIGHT, FRAME_WIDTH, EMBEDDING_DIM,
    BLOB_SIGMA, BLOB_VELOCITY, NUM_SEQUENCES, SEQUENCE_LENGTH,
)
from .world import render_blob, make_sequences, roll_one, build_pairs
from .models import Encoder, Predictor, Decoder, MLPBaseline, variance_term
from .train import (
    train_jepa,
    train_decoder,
    train_baseline,
    linear_probe,
    forecast,
    render_dream_gif,
)

__all__ = [
    # config
    "FRAME_HEIGHT", "FRAME_WIDTH", "EMBEDDING_DIM",
    "BLOB_SIGMA", "BLOB_VELOCITY", "NUM_SEQUENCES", "SEQUENCE_LENGTH",
    # world
    "render_blob", "make_sequences", "roll_one", "build_pairs",
    # models
    "Encoder", "Predictor", "Decoder", "MLPBaseline", "variance_term",
    # training & evaluation
    "train_jepa", "train_decoder", "train_baseline",
    "linear_probe", "forecast", "render_dream_gif",
]
__version__ = "0.1.0"
