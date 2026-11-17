import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_fnirs_data import simulate
from src.train_and_evaluate import train_and_evaluate


def test_at_least_1200_subjects():
    features, age = simulate()
    assert len(features) >= 1200


def test_mae_is_reasonable():
    features, age = simulate()
    result = train_and_evaluate(features, age)
    assert result["mae_years"] < 15.0


def test_attention_recovers_most_true_channels():
    features, age = simulate()
    result = train_and_evaluate(features, age)
    assert result["recovery_pct"] >= 50.0


def test_top8_attended_channels_is_exactly_8():
    features, age = simulate()
    result = train_and_evaluate(features, age)
    assert len(result["top8_attended_channels"]) == 8


def test_age_distribution_is_realistic():
    features, age = simulate()
    assert age.min() >= 18 and age.max() <= 85
