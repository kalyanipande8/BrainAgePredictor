"""Simulate fNIRS-style hemodynamic signal features for 1,220
subjects across 16 cortical-region channels, where a known true
subset of 8 channels' signal features genuinely drive age (the rest
are noise channels) -- the ground truth the attention visualization
module must recover."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_SUBJECTS = 1220
N_CHANNELS = 16
SEED = 1091
TRUE_PREDICTIVE_CHANNELS = [1, 3, 4, 7, 9, 11, 13, 14]  # 8 of 16 channels genuinely age-related


def simulate(n_subjects: int = N_SUBJECTS, n_channels: int = N_CHANNELS, seed: int = SEED):
    rng = np.random.default_rng(seed)

    age = rng.uniform(18, 85, n_subjects)
    channel_features = rng.normal(0, 1, (n_subjects, n_channels))

    age_z = (age - age.mean()) / age.std()
    for ch in TRUE_PREDICTIVE_CHANNELS:
        weight = rng.uniform(2.5, 4.0)
        channel_features[:, ch] += weight * age_z

    return channel_features.astype(np.float32), age.astype(np.float32)


if __name__ == "__main__":
    features, age = simulate()
    print(features.shape, age.mean(), age.std())
