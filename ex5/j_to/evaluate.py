"""Evaluate the fan anomaly detector on the evaluation dataset."""

from pathlib import Path

from cnn import THRESHOLD, predict
from pandas import read_csv
from sklearn.metrics import classification_report, f1_score

DATA_DIR = Path("data/eval")
INDICES = ["03", "04", "05", "06"]


def evaluate():
    """Run inference on evaluation data and compute F1 scores."""
    raw = read_csv("data/eval_mapping.csv")
    laballed = dict(zip(list(raw["eval_filename"]), list(raw["condition"])))

    f1_final_denom = 0.0
    for index in INDICES:
        files = f"model_{index}_*.wav"
        paths = sorted(DATA_DIR.glob(files))
        if not paths:
            raise FileNotFoundError(f"No audio files found for {files}")

        labels, predictions = [], []
        for path in paths:
            probability = predict(str(path))
            label = int("anomaly" in laballed[path.name])
            labels.append(label)
            prediction = int(probability > THRESHOLD)
            predictions.append(prediction)
            if prediction != label:
                print(
                    f"{path.name}, {probability:.4f}, {"anomaly" if label else "normal"}"
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
