"""Run the full DVD-JEPA training pipeline and save all outputs.

Usage:
    python scripts/train.py

Outputs:
    checkpoints/dvd_jepa.pt         PyTorch weights for all models
    assets/dvd_jepa_dream.gif       3-panel GIF: ground truth | JEPA | MLP baseline
    assets/metrics.json             Forecast MSE for both models + probe RMSE
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import dvd_jepa as dj
from dvd_jepa.config import (
    BLOB_VELOCITY,
    NUM_SEQUENCES, SEQUENCE_LENGTH, NOISE_STD_DEV,
    JEPA_TRAIN_STEPS, DECODER_TRAIN_STEPS, BASELINE_TRAIN_STEPS, PROBE_TRAIN_STEPS,
    BATCH_SIZE, LEARNING_RATE,
    FORECAST_HORIZON, ANOMALY_SEQUENCE_LENGTH,
    CHECKPOINT_DIR, ASSETS_DIR,
)

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    print(f"DVD-JEPA :: training on {DEVICE}\n")

    # Generate training data
    frames, positions = dj.make_sequences(
        num_sequences=NUM_SEQUENCES,
        sequence_length=SEQUENCE_LENGTH,
        velocity=BLOB_VELOCITY,
        noise_std_dev=NOISE_STD_DEV,
        seed=0,
    )
    obs, next_obs = dj.build_pairs(frames)

    # Position labels: for each obs pair (frame_t, frame_{t+1}), label is position at t+1.
    position_labels = torch.tensor(
        np.concatenate([positions[:, t + 1] for t in range(SEQUENCE_LENGTH - 2)]),
        dtype=torch.float32,
    )

    # Train JEPA: online encoder + predictor via EMA target encoder
    print("--- JEPA ---")
    online_encoder, target_encoder, predictor = dj.train_jepa(
        obs, next_obs,
        steps=JEPA_TRAIN_STEPS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        device=DEVICE,
    )

    # Train decoder: reconstruct frames from frozen target encoder embeddings
    print("--- Decoder ---")
    decoder = dj.train_decoder(
        target_encoder, obs,
        steps=DECODER_TRAIN_STEPS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
    )

    # Train MLP baseline: direct pixel-to-pixel prediction from two stacked frames
    print("--- MLP Baseline ---")
    baseline = dj.train_baseline(
        obs,
        steps=BASELINE_TRAIN_STEPS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        device=DEVICE,
    )

    # Linear probe: does the latent encode position without ever seeing coordinates?
    print("--- Linear probe ---")
    _, probe_rmse = dj.linear_probe(
        target_encoder, obs, position_labels,
        steps=PROBE_TRAIN_STEPS,
    )

    # Forecast comparison: roll both models forward from the same seed video
    print("--- Forecast ---")
    eval_frames, _ = dj.roll_one(
        sequence_length=ANOMALY_SEQUENCE_LENGTH, seed=7, velocity=BLOB_VELOCITY,
    )
    jepa_predictions, baseline_predictions, jepa_errors, baseline_errors = dj.forecast(
        target_encoder, predictor, decoder, baseline,
        eval_frames, horizon=FORECAST_HORIZON,
    )
    ground_truth_frames = [eval_frames[k + 2] for k in range(FORECAST_HORIZON)]

    # Save checkpoint
    checkpoint_path = ROOT / CHECKPOINT_DIR / "dvd_jepa.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "online_encoder": online_encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "predictor":      predictor.state_dict(),
            "decoder":        decoder.state_dict(),
            "baseline":       baseline.state_dict(),
        },
        checkpoint_path,
    )

    # Render three-panel comparison GIF
    gif_path = ROOT / ASSETS_DIR / "dvd_jepa_dream.gif"
    dj.render_dream_gif(ground_truth_frames, jepa_predictions, baseline_predictions, gif_path)

    # Save metrics
    metrics = {
        "probe_position_rmse_px":          round(probe_rmse, 3),
        "jepa_forecast_mse_step_1":        round(jepa_errors[0], 5),
        "jepa_forecast_mse_step_final":    round(jepa_errors[-1], 5),
        "jepa_forecast_mse_mean":          round(float(np.mean(jepa_errors)), 5),
        "baseline_forecast_mse_step_1":    round(baseline_errors[0], 5),
        "baseline_forecast_mse_step_final": round(baseline_errors[-1], 5),
        "baseline_forecast_mse_mean":      round(float(np.mean(baseline_errors)), 5),
        "forecast_horizon":                FORECAST_HORIZON,
    }
    metrics_path = ROOT / ASSETS_DIR / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("\n--- Results ---")
    for key, value in metrics.items():
        print(f"  {key:<42} {value}")
    print(f"\nWrote: {checkpoint_path.relative_to(ROOT)}")
    print(f"       {gif_path.relative_to(ROOT)}")
    print(f"       {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
