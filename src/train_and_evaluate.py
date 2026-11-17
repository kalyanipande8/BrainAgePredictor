"""Real training loop (Adam, MSE loss) for the attention-based brain-
age regressor, with held-out-test MAE evaluation and real attention-
weight extraction validated against the simulation's true predictive
channels."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

from .attention_model import AttentionAgeRegressor
from .simulate_fnirs_data import TRUE_PREDICTIVE_CHANNELS


def train_and_evaluate(features: np.ndarray, age: np.ndarray, test_frac: float = 0.2,
                        epochs: int = 200, seed: int = 1091) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = len(features)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_train = torch.tensor(features[train_idx])
    y_train = torch.tensor(age[train_idx])
    X_test = torch.tensor(features[test_idx])
    y_test = torch.tensor(age[test_idx])

    model = AttentionAgeRegressor()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.MSELoss()

    batch_size = 64
    n_train = len(X_train)
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx_b = perm[i:i + batch_size]
            optimizer.zero_grad()
            pred = model(X_train[idx_b])
            loss = criterion(pred, y_train[idx_b])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred, attn_weights = model(X_test, return_attention=True)

    mae = mean_absolute_error(y_test.numpy(), test_pred.numpy())
    mean_attn_per_channel = attn_weights.mean(dim=0).numpy()

    top8_channels = set(np.argsort(mean_attn_per_channel)[::-1][:8].tolist())
    true_channels = set(TRUE_PREDICTIVE_CHANNELS)
    recovered = top8_channels & true_channels

    return {"n_train": n_train, "n_test": len(test_idx), "mae_years": round(float(mae), 3),
            "top8_attended_channels": sorted(top8_channels),
            "true_predictive_channels": sorted(true_channels),
            "n_recovered": len(recovered), "recovery_pct": round(100 * len(recovered) / 8, 2)}
