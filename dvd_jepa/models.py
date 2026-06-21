"""The four networks that make up DVD-JEPA and its baseline comparison.

    Encoder      obs (2*H*W)  -> z (EMBEDDING_DIM)
    Predictor    z (EMB)      -> z_hat (EMB)         the world model / transition fn
    Decoder      z (EMB)      -> frame (H*W)          optional pixel readout head
    MLPBaseline  obs (2*H*W)  -> frame (H*W)          direct pixel-space baseline

The Encoder is used twice: once as the trainable online encoder and once as an
Exponential-Moving-Average target encoder whose output is the prediction target
(stop-gradient). That asymmetry, plus the variance term below, is what stops the
representation from collapsing to a constant.

MLPBaseline has a comparable parameter count to the active JEPA system
(online encoder + predictor + decoder) so the comparison is fair: same
inductive bias (pure MLP, GELU), same parameter budget, same input format.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    FRAME_HEIGHT, FRAME_WIDTH, EMBEDDING_DIM,
    ENCODER_HIDDEN_DIMS, PREDICTOR_HIDDEN_DIM, DECODER_HIDDEN_DIMS, BASELINE_HIDDEN_DIM,
)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_1, hidden_2 = ENCODER_HIDDEN_DIMS
        self.net = nn.Sequential(
            nn.Linear(2 * FRAME_HEIGHT * FRAME_WIDTH, hidden_1), nn.GELU(),
            nn.Linear(hidden_1, hidden_2), nn.GELU(),
            nn.Linear(hidden_2, EMBEDDING_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """The world model proper: advance one step in representation space."""

    def __init__(self):
        super().__init__()
        hidden = PREDICTOR_HIDDEN_DIM
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, EMBEDDING_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Decoder(nn.Module):
    """Turn a latent back into a pixel frame. A pure JEPA has no decoder; this
    is what makes the dream visible and measurable."""

    def __init__(self):
        super().__init__()
        hidden_1, hidden_2 = DECODER_HIDDEN_DIMS
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, hidden_1), nn.GELU(),
            nn.Linear(hidden_1, hidden_2), nn.GELU(),
            nn.Linear(hidden_2, FRAME_HEIGHT * FRAME_WIDTH), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MLPBaseline(nn.Module):
    """Pixel-space MLP baseline: predict the next frame directly from two stacked frames.

    Input:  2 * FRAME_HEIGHT * FRAME_WIDTH  — two consecutive frames, flattened.
    Output: FRAME_HEIGHT * FRAME_WIDTH      — predicted next frame pixels (sigmoid).

    Parameter count (~262k) is within 2% of the active JEPA system (online encoder
    + predictor + decoder, ~257k), so the comparison is apples-to-apples.
    """

    def __init__(self):
        super().__init__()
        hidden = BASELINE_HIDDEN_DIM
        self.net = nn.Sequential(
            nn.Linear(2 * FRAME_HEIGHT * FRAME_WIDTH, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, FRAME_HEIGHT * FRAME_WIDTH), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def variance_term(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style hinge: encourage every embedding dimension to keep a standard
    deviation of at least 1 across the batch. Explicit anti-collapse pressure."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()
