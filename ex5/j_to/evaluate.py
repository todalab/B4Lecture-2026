"""Evaluate the fan anomaly detector on the evaluation dataset."""

from pathlib import Path

from pandas import read_csv
from sklearn.metrics import classification_report, f1_score

from cnn import THRESHOLD, predict

DATA_DIR = Path("data/eval")
INDICES = ["03", "04", "05", "06"]


def evaluate():
    """Run inference on evaluation data and compute F1 scores."""
    raw = read_csv("data/eval_mapping.csv")
    laballed = dict(zip(list(raw["eval_filename"]), list(raw["condition"])))

    f1_final_denom = 0.0
    for index in INDICES:
        paths = sorted(DATA_DIR.glob(f"model_{index}_*.wav"))
        if not paths:
            exit(1)

        labels, predictions = [], []
        for path in paths:
            probablity = predict(str(path))
            label = int("anomaly" in laballed[path.name])
            labels.append(label)
            prediction = int(probablity > THRESHOLD)
            predictions.append(prediction)
            if prediction != label:
                print(
                    f"{path.name}, {probablity:.4f}, {"anomaly" if label else "normal"}"
                )

        f1 = f1_score(labels, predictions)
        f1_final_denom += 1 / f1
        print(f"model #{index}, f1: {f1}")
        print(
            classification_report(
                labels,
                predictions,
                target_names=["normal", "anomaly"],
            )
        )
    f1_final = 4 / f1_final_denom
    print(f"f1_final: {f1_final}")


if __name__ == "__main__":
    evaluate()
