"""正常データと異常データのデータリストのテキストファイルを生成する."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path


def make_normal_and_anomaly_data_list(seed: int = 42) -> None:
    """正常データと異常データの学習用・評価用・テスト用のデータリストを生成する.

    各モデルごとにファイルをシャッフルし、8:1:1 の割合で
    normal_and_anomaly_data_train.txt, normal_and_anomaly_data_eval.txt,
    normal_and_anomaly_data_test.txt をこのファイルと同じフォルダに出力する.
    ファイル内には data/dev からの相対パスを1行ずつ書き出す.
    """
    script_dir = Path(__file__).resolve().parent
    data_dir = Path(__file__).resolve().parents[3] / "data" / "dev"

    normal_files_by_model: dict[str, list[Path]] = defaultdict(list)
    anomaly_files_by_model: dict[str, list[Path]] = defaultdict(list)
    # 正常データ（model_xx_normal_xxxxxxxx.wav）と異常データ（model_xx_anomaly_xxxxxxxx）を均等に分ける.
    for wav_path in sorted(data_dir.glob("*_normal_*.wav")):
        parts = wav_path.stem.split("_")
        if len(parts) < 2:
            continue
        model_name = "_".join(parts[:2])
        normal_files_by_model[model_name].append(wav_path)

    for wav_path in sorted(data_dir.glob("*_anomaly_*.wav")):
        parts = wav_path.stem.split("_")
        if len(parts) < 2:
            continue
        model_name = "_".join(parts[:2])
        anomaly_files_by_model[model_name].append(wav_path)

    rng = random.Random(seed)
    train_files: list[Path] = []
    eval_files: list[Path] = []
    test_files: list[Path] = []

    for model_name in sorted(normal_files_by_model):
        model_files = normal_files_by_model[model_name]
        rng.shuffle(model_files)
        total = len(model_files)
        train_count = total * 8 // 10
        eval_count = total * 1 // 10
        test_count = total - train_count - eval_count

        train_files.extend(model_files[:train_count])
        eval_files.extend(model_files[train_count : train_count + eval_count])
        test_files.extend(model_files[train_count + eval_count : train_count + eval_count + test_count])

    for model_name in sorted(anomaly_files_by_model):
        model_files = anomaly_files_by_model[model_name]
        rng.shuffle(model_files)
        total = len(model_files)
        train_count = total * 8 // 10
        eval_count = total * 1 // 10
        test_count = total - train_count - eval_count

        train_files.extend(model_files[:train_count])
        eval_files.extend(model_files[train_count : train_count + eval_count])
        test_files.extend(model_files[train_count + eval_count : train_count + eval_count + test_count])

    def write_list(file_path: Path, entries: list[Path]) -> None:
        with file_path.open("w", encoding="utf-8") as f:
            for wav_path in entries:
                f.write(f"{wav_path.relative_to(data_dir)}\n")

    write_list(script_dir / "normal_and_anomaly_data_train.txt", train_files)
    write_list(script_dir / "normal_and_anomaly_data_eval.txt", eval_files)
    write_list(script_dir / "normal_and_anomaly_data_test.txt", test_files)


if __name__ == "__main__":
    make_normal_and_anomaly_data_list()
