from pathlib import Path

from sklearn.metrics import classification_report, f1_score
from main import predict, THRESHOLD

DATA_DIR = Path("data/dev")
MODEL_IDS = ["03", "04", "05", "06"]


def evaluate():
    for model_id in MODEL_IDS:
        paths = sorted(DATA_DIR.glob(f"model_{model_id}_*.wav"))
        if not paths:
            exit(1)

        labels, predictions = [], []
        for path in paths:
            probablity = predict(str(path))
            label = int("anomaly" in path.name)
            labels.append(label)
            prediction = int(probablity > THRESHOLD)
            predictions.append(prediction)
            if prediction != label:
                print(path)

        n_anomaly = sum(labels)
        f1 = f1_score(labels, predictions)
        print(
            f"model #{model_id}\n"
            f"n: {len(labels)}, anomaly: {n_anomaly}, normal: {len(labels) - n_anomaly}\n"
            f"F1 score: {f1:.4f}\n"
        )
        print(
            classification_report(
                labels,
                predictions,
                target_names=["normal", "anomaly"],
            )
        )


if __name__ == "__main__":
    evaluate()
