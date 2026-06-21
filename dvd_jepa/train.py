"""Training functions and evaluation utilities for DVD-JEPA.

This module is a library of pure functions. The entry point that orchestrates
a full training run lives in scripts/train.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    FRAME_HEIGHT, FRAME_WIDTH, EMBEDDING_DIM,
    EMA_DECAY, TRAIN_FRACTION,
    JEPA_TRAIN_STEPS, DECODER_TRAIN_STEPS, BASELINE_TRAIN_STEPS, PROBE_TRAIN_STEPS,
    BATCH_SIZE, LEARNING_RATE, PROBE_LEARNING_RATE,
    FORECAST_HORIZON,
)
from .models import Encoder, Predictor, Decoder, MLPBaseline, variance_term


def _resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_jepa(
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    steps: int = JEPA_TRAIN_STEPS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    log=print,
    device: torch.device | None = None,
) -> tuple[Encoder, Encoder, Predictor]:
    """Train the JEPA encoder and predictor with an EMA target encoder.

    Returns the online encoder, target encoder, and predictor.
    """
    device = _resolve_device(device)
    online = Encoder().to(device)
    target = Encoder().to(device)
    predictor = Predictor().to(device)
    target.load_state_dict(online.state_dict())
    for param in target.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(
        list(online.parameters()) + list(predictor.parameters()), lr=learning_rate
    )
    obs, next_obs = obs.to(device), next_obs.to(device)
    num_samples = obs.shape[0]
    for step in range(1, steps + 1):
        idx = torch.randint(0, num_samples, (batch_size,), device=device)
        z_online = online(obs[idx])
        with torch.no_grad():
            z_target = target(next_obs[idx])
        loss = F.mse_loss(predictor(z_online), z_target) + variance_term(z_online)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for p_online, p_target in zip(online.parameters(), target.parameters()):
                p_target.mul_(EMA_DECAY).add_(p_online, alpha=1 - EMA_DECAY)
        if log and (step % 500 == 0):
            log(f"  jepa step {step:4d} | loss {loss.item():.4f} | emb-std {z_online.std(0).mean():.2f}")
    return online, target, predictor


def train_decoder(
    target: Encoder,
    obs: torch.Tensor,
    steps: int = DECODER_TRAIN_STEPS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    log=print,
) -> Decoder:
    """Train a decoder to reconstruct frames from frozen target encoder embeddings."""
    device = next(target.parameters()).device
    decoder = Decoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    obs = obs.to(device)
    with torch.no_grad():
        embeddings = target(obs)
    target_frames = obs[:, FRAME_HEIGHT * FRAME_WIDTH:]
    num_samples = obs.shape[0]
    final_loss = torch.tensor(0.0)
    for step in range(1, steps + 1):
        idx = torch.randint(0, num_samples, (batch_size,), device=device)
        final_loss = F.mse_loss(decoder(embeddings[idx]), target_frames[idx])
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
    if log:
        log(f"  decoder recon MSE {final_loss.item():.5f}")
    return decoder


def train_baseline(
    obs: torch.Tensor,
    steps: int = BASELINE_TRAIN_STEPS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    log=print,
    device: torch.device | None = None,
) -> MLPBaseline:
    """Train the MLP baseline to predict the next frame directly in pixel space.

    Input:  two consecutive frames stacked (2 * FRAME_HEIGHT * FRAME_WIDTH).
    Target: the second (newer) frame in each observation pair.
    """
    device = _resolve_device(device)
    baseline = MLPBaseline().to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=learning_rate)
    obs = obs.to(device)
    target_frames = obs[:, FRAME_HEIGHT * FRAME_WIDTH:]
    num_samples = obs.shape[0]
    final_loss = torch.tensor(0.0)
    for step in range(1, steps + 1):
        idx = torch.randint(0, num_samples, (batch_size,), device=device)
        final_loss = F.mse_loss(baseline(obs[idx]), target_frames[idx])
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        if log and (step % 500 == 0):
            log(f"  baseline recon MSE {final_loss.item():.5f}")
    return baseline


def linear_probe(
    target: Encoder,
    obs: torch.Tensor,
    position_labels: torch.Tensor,
    steps: int = PROBE_TRAIN_STEPS,
    log=print,
) -> tuple[torch.nn.Linear, float]:
    """Frozen-encoder linear readout of ground-truth (y, x) position.

    Proves the latent encodes world state even though it was never given
    coordinates during training.
    """
    device = next(target.parameters()).device
    obs, position_labels = obs.to(device), position_labels.to(device)
    with torch.no_grad():
        embeddings = target(obs)
    num_samples = embeddings.shape[0]
    num_train = int(TRAIN_FRACTION * num_samples)
    probe = torch.nn.Linear(EMBEDDING_DIM, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LEARNING_RATE)
    for _ in range(steps):
        idx = torch.randint(0, num_train, (BATCH_SIZE,), device=device)
        loss = F.mse_loss(probe(embeddings[idx]), position_labels[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        rmse = torch.sqrt(
            F.mse_loss(probe(embeddings[num_train:]), position_labels[num_train:])
        ).item()
    if log:
        log(f"  linear probe position RMSE {rmse:.2f} px")
    return probe, rmse


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def forecast(
    target: Encoder,
    predictor: Predictor,
    decoder: Decoder,
    baseline: MLPBaseline,
    video_frames: np.ndarray,
    horizon: int = FORECAST_HORIZON,
) -> tuple[list, list, list, list]:
    """Roll both models forward for `horizon` steps from the same seed frames.

    The JEPA rolls in latent space; the baseline rolls autoregressively in
    pixel space, feeding each predicted frame back as input for the next step.

    Returns
    -------
    jepa_predictions     : list of `horizon` arrays, shape [FRAME_HEIGHT, FRAME_WIDTH]
    baseline_predictions : list of `horizon` arrays, shape [FRAME_HEIGHT, FRAME_WIDTH]
    jepa_errors          : list of `horizon` per-step MSE values vs ground truth
    baseline_errors      : list of `horizon` per-step MSE values vs ground truth
    """
    device = next(target.parameters()).device
    with torch.no_grad():
        seed = torch.tensor(
            video_frames[0:2].reshape(1, -1), dtype=torch.float32, device=device
        )
        z = target(seed)
        jepa_predictions, jepa_errors = [], []
        for k in range(horizon):
            z = predictor(z)
            frame = decoder(z).reshape(FRAME_HEIGHT, FRAME_WIDTH).cpu().numpy()
            jepa_predictions.append(frame)
            jepa_errors.append(float(((frame - video_frames[k + 2]) ** 2).mean()))

        baseline = baseline.to(device)
        frame_a, frame_b = video_frames[0], video_frames[1]
        baseline_predictions, baseline_errors = [], []
        for k in range(horizon):
            inp = torch.tensor(
                np.stack([frame_a, frame_b]).reshape(1, -1),
                dtype=torch.float32, device=device,
            )
            predicted_frame = baseline(inp).reshape(FRAME_HEIGHT, FRAME_WIDTH).cpu().numpy()
            baseline_predictions.append(predicted_frame)
            baseline_errors.append(float(((predicted_frame - video_frames[k + 2]) ** 2).mean()))
            frame_a, frame_b = frame_b, predicted_frame

    return jepa_predictions, baseline_predictions, jepa_errors, baseline_errors


# ---------------------------------------------------------------------------
# Asset rendering
# ---------------------------------------------------------------------------

def render_dream_gif(
    ground_truth_frames: list,
    jepa_frames: list,
    baseline_frames: list,
    path,
    scale: int = 9,
) -> None:
    """Render a three-panel GIF: ground truth | JEPA dream | MLP baseline dream."""
    from PIL import Image, ImageDraw

    gap = 10
    label_bar = 18
    panel_width = FRAME_WIDTH * scale
    panel_height = FRAME_HEIGHT * scale

    def to_rgb(frame, color):
        a = np.clip(frame, 0, 1)
        rgb = (np.stack([a * color[0], a * color[1], a * color[2]], axis=-1) * 255).astype(np.uint8)
        return Image.fromarray(rgb, "RGB").resize((panel_width, panel_height), Image.NEAREST)

    gif_frames = []
    for k in range(len(jepa_frames)):
        canvas = Image.new(
            "RGB",
            (panel_width * 3 + gap * 2, panel_height + label_bar),
            (18, 18, 22),
        )
        canvas.paste(to_rgb(ground_truth_frames[k], (1.0, 1.0, 1.0)), (0, label_bar))
        canvas.paste(to_rgb(jepa_frames[k], (0.3, 0.9, 1.0)), (panel_width + gap, label_bar))
        canvas.paste(to_rgb(baseline_frames[k], (1.0, 0.75, 0.2)), (2 * (panel_width + gap), label_bar))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), "ground truth", fill=(170, 170, 175))
        draw.text((panel_width + gap + 4, 4), "JEPA dream", fill=(90, 210, 255))
        draw.text((2 * (panel_width + gap) + 4, 4), "MLP baseline", fill=(255, 190, 50))
        draw.text((4, panel_height + label_bar - 12), f"t+{k + 1}", fill=(110, 110, 120))
        gif_frames.append(canvas)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:], duration=130, loop=0)
