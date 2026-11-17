import json
import os

from src.simulate_fnirs_data import simulate
from src.train_and_evaluate import train_and_evaluate


def main():
    features, age = simulate()
    result = train_and_evaluate(features, age)

    summary = {"n_subjects_total": len(features), **result}

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
