"""The world: a DVD logo bouncing in a box.

There are no actions and no agent. The logo moves at constant speed and
reflects off the walls. This is deliberately the simplest non-trivial dynamical
system that still has the property that matters for a world model: the future
is not readable from a single frame (you need two frames to know the velocity),
but it is perfectly predictable once you do.

A single rendered "observation" handed to the model is a stack of two
consecutive frames, so velocity is observable.
"""
from __future__ import annotations

import numpy as np

from .config import (
    FRAME_HEIGHT, FRAME_WIDTH, BLOB_SIGMA,
    SPAWN_MARGIN, WALL_LOW, WALL_HIGH,
    NUM_SEQUENCES, SEQUENCE_LENGTH, BLOB_VELOCITY, NOISE_STD_DEV,
)

# Pixel-coordinate grid, computed once and reused in all rendering calls.
_YY, _XX = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]


def render_blob(position: np.ndarray) -> np.ndarray:
    """Render a soft Gaussian blob at position (y, x) into a FRAME_HEIGHT x FRAME_WIDTH frame."""
    dy = _YY - position[0]
    dx = _XX - position[1]
    return np.exp(-((dy**2 + dx**2) / (2 * BLOB_SIGMA**2))).astype(np.float32)


def _physics_step(
    position: np.ndarray, velocity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Advance position by velocity and reflect off walls.

    Handles both batched input [num_sequences, 2] and scalar input [2].
    """
    position = position + velocity
    for dim in range(2):
        if position.ndim == 2:
            outside = (position[:, dim] > WALL_HIGH) | (position[:, dim] < WALL_LOW)
            velocity[outside, dim] *= -1
            position[:, dim] = np.clip(position[:, dim], WALL_LOW, WALL_HIGH)
        else:
            if position[dim] > WALL_HIGH or position[dim] < WALL_LOW:
                velocity[dim] *= -1
            position[dim] = np.clip(position[dim], WALL_LOW, WALL_HIGH)
    return position, velocity


def make_sequences(
    num_sequences: int = NUM_SEQUENCES,
    sequence_length: int = SEQUENCE_LENGTH,
    velocity: float = BLOB_VELOCITY,
    seed: int | None = None,
    noise_std_dev: float = NOISE_STD_DEV,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a batch of bouncing-logo videos.

    Returns
    -------
    frames    : float32 [num_sequences, sequence_length, FRAME_HEIGHT, FRAME_WIDTH]
    positions : float32 [num_sequences, sequence_length, 2]  ground-truth (y, x) centre
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    positions = np.zeros((num_sequences, sequence_length, 2), np.float32)
    frames = np.zeros((num_sequences, sequence_length, FRAME_HEIGHT, FRAME_WIDTH), np.float32)
    p = rng.uniform(SPAWN_MARGIN, FRAME_HEIGHT - SPAWN_MARGIN, size=(num_sequences, 2)).astype(np.float32)
    vel = (rng.choice([-1, 1], size=(num_sequences, 2)) * velocity * (FRAME_HEIGHT - 1)).astype(np.float32)
    for t in range(sequence_length):
        p, vel = _physics_step(p, vel)
        positions[:, t] = p
        centre_y = p[:, 0][:, None, None]
        centre_x = p[:, 1][:, None, None]
        frames[:, t] = np.exp(-(
            ((_YY[None] - centre_y)**2 + (_XX[None] - centre_x)**2) / (2 * BLOB_SIGMA**2)
        ))
    if noise_std_dev > 0.0:
        frames = np.clip(
            frames + rng.normal(0.0, noise_std_dev, frames.shape).astype(np.float32),
            0.0, 1.0,
        )
    return frames, positions


def roll_one(
    sequence_length: int = SEQUENCE_LENGTH,
    teleport_at: int | None = None,
    seed: int = 0,
    velocity: float = BLOB_VELOCITY,
    noise_std_dev: float = NOISE_STD_DEV,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single video; optionally teleport the logo at one step to inject
    an anomaly the world model has no way to anticipate."""
    rng = np.random.RandomState(seed)
    p = rng.uniform(SPAWN_MARGIN, FRAME_HEIGHT - SPAWN_MARGIN, size=2).astype(np.float32)
    vel = (rng.choice([-1, 1], size=2) * velocity * (FRAME_HEIGHT - 1)).astype(np.float32)
    frames = np.zeros((sequence_length, FRAME_HEIGHT, FRAME_WIDTH), np.float32)
    positions = np.zeros((sequence_length, 2), np.float32)
    for t in range(sequence_length):
        if teleport_at is not None and t == teleport_at:
            p = rng.uniform(SPAWN_MARGIN, FRAME_HEIGHT - SPAWN_MARGIN, size=2).astype(np.float32)
        p, vel = _physics_step(p, vel)
        positions[t] = p
        frames[t] = render_blob(p)
        if noise_std_dev > 0.0:
            frames[t] = np.clip(
                frames[t] + rng.normal(0.0, noise_std_dev, frames[t].shape).astype(np.float32),
                0.0, 1.0,
            )
    return frames, positions


def build_pairs(frames: np.ndarray) -> tuple:
    """Convert a batch of videos into consecutive observation pairs.

    Each observation is two consecutive frames flattened and concatenated
    (shape 2 * FRAME_HEIGHT * FRAME_WIDTH). The model learns to predict the
    embedding of the next observation from the current one. The second half of
    each observation vector is the middle frame, which is what the decoder
    and baseline are trained to reconstruct.

    Parameters
    ----------
    frames : float32 [num_sequences, sequence_length, FRAME_HEIGHT, FRAME_WIDTH]

    Returns
    -------
    obs      : float32 [N, 2*FRAME_HEIGHT*FRAME_WIDTH]  current observations
    next_obs : float32 [N, 2*FRAME_HEIGHT*FRAME_WIDTH]  next observations
    """
    import torch

    num_sequences, sequence_length = frames.shape[0], frames.shape[1]
    obs_list, next_obs_list = [], []
    for t in range(sequence_length - 2):
        obs_list.append(frames[:, t:t + 2].reshape(num_sequences, -1))
        next_obs_list.append(frames[:, t + 1:t + 3].reshape(num_sequences, -1))
    obs = torch.tensor(np.concatenate(obs_list), dtype=torch.float32)
    next_obs = torch.tensor(np.concatenate(next_obs_list), dtype=torch.float32)
    return obs, next_obs
