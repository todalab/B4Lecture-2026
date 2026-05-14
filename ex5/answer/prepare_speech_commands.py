import argparse
import io
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

MAIN_DATA_URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
TEST_DATA_URL = "https://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"


DEFAULT_TARGET_LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]


def normalize_member_name(name):
    return name[2:] if name.startswith("./") else name


def find_tarballs(download_dir):
    download_dir.mkdir(parents=True, exist_ok=True)
    tarballs = sorted(download_dir.glob("*.tar.gz"))
    if len(tarballs) < 2:
        download_file(MAIN_DATA_URL, download_dir / "speech_commands_v0.02.tar.gz")
        download_file(TEST_DATA_URL, download_dir / "speech_commands_test_set_v0.02.tar.gz")
        tarballs = sorted(download_dir.glob("*.tar.gz"))

    if len(tarballs) < 2:
        raise FileNotFoundError(
            f"expected cached Speech Commands tarballs under {download_dir}, found {len(tarballs)}"
        )

    main_tar = None
    test_tar = None
    for path in tarballs:
        name = path.name
        if "test_set" in name or "comm_test_set" in name:
            test_tar = path
        elif "speech_command" in name or "speech_comm" in name:
            main_tar = path

    if main_tar is None or test_tar is None:
        raise FileNotFoundError(f"could not identify main/test tarballs in {download_dir}")
    return main_tar, test_tar


def download_file(url, output):
    if output.exists():
        return
    print(f"downloading: {url}")
    print(f"        to: {output}")
    urllib.request.urlretrieve(url, output)


def read_text_member(tar, member_name):
    member = tar.getmember(member_name)
    with tar.extractfile(member) as f:
        return f.read().decode("utf-8")


def read_split_lists(main_tar_path):
    with tarfile.open(main_tar_path, "r:gz") as tar:
        validation = {
            line.strip()
            for line in read_text_member(tar, "./validation_list.txt").splitlines()
            if line.strip()
        }
        testing = {
            line.strip()
            for line in read_text_member(tar, "./testing_list.txt").splitlines()
            if line.strip()
        }
    return validation, testing


def read_wav_from_tar(tar, member, audio_length):
    with tar.extractfile(member) as f:
        wav_bytes = f.read()
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) >= audio_length:
        return audio[:audio_length]
    return np.pad(audio, (0, audio_length - len(audio)))


def empty_counts(target_labels):
    return {label: 0 for label in target_labels}


def collect_from_main_tar(main_tar_path, target_labels, split_name, max_per_class, audio_length):
    validation_list, testing_list = read_split_lists(main_tar_path)
    counts = empty_counts(target_labels)
    target_set = set(target_labels)
    X = []
    y = []

    with tarfile.open(main_tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".wav"):
                continue

            name = normalize_member_name(member.name)
            parts = Path(name).parts
            if len(parts) != 2:
                continue
            label = parts[0]
            if label not in target_set:
                continue

            in_validation = name in validation_list
            in_testing = name in testing_list
            if split_name == "train" and (in_validation or in_testing):
                continue
            if split_name == "validation" and not in_validation:
                continue

            if counts[label] >= max_per_class:
                continue

            X.append(read_wav_from_tar(tar, member, audio_length))
            y.append(target_labels.index(label))
            counts[label] += 1

            if all(count >= max_per_class for count in counts.values()):
                break

    return counts, np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def collect_from_test_tar(test_tar_path, target_labels, max_per_class, audio_length):
    counts = empty_counts(target_labels)
    target_set = set(target_labels)
    X = []
    y = []

    with tarfile.open(test_tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".wav"):
                continue

            name = normalize_member_name(member.name)
            parts = Path(name).parts
            if len(parts) != 2:
                continue
            label = parts[0]
            if label not in target_set or counts[label] >= max_per_class:
                continue

            X.append(read_wav_from_tar(tar, member, audio_length))
            y.append(target_labels.index(label))
            counts[label] += 1

            if all(count >= max_per_class for count in counts.values()):
                break

    return counts, np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def validate_counts(split_name, counts, required):
    missing = [label for label, count in counts.items() if count < required]
    if missing:
        raise RuntimeError(f"{split_name}: insufficient examples for labels {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="answer/data/speech_commands_subset.npz")
    parser.add_argument("--data-dir", default="answer/tfds_data")
    parser.add_argument("--labels", default=",".join(DEFAULT_TARGET_LABELS))
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--audio-length", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    target_labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    download_dir = Path(args.data_dir) / "downloads" / "speech_commands"
    main_tar, test_tar = find_tarballs(download_dir)
    rng = np.random.default_rng(args.seed)

    train_counts, X_train, y_train = collect_from_main_tar(
        main_tar, target_labels, "train", args.train_per_class, args.audio_length
    )
    validate_counts("train", train_counts, args.train_per_class)

    validation_counts, X_validation, y_validation = collect_from_main_tar(
        main_tar, target_labels, "validation", args.validation_per_class, args.audio_length
    )
    validate_counts("validation", validation_counts, args.validation_per_class)

    test_counts, X_test, y_test = collect_from_test_tar(
        test_tar, target_labels, args.test_per_class, args.audio_length
    )
    validate_counts("test", test_counts, args.test_per_class)

    split_data = [
        ("train", X_train, y_train),
        ("validation", X_validation, y_validation),
        ("test", X_test, y_test),
    ]

    X_parts = []
    y_parts = []
    split_parts = []
    for split_name, X_split, y_split in split_data:
        order = rng.permutation(len(y_split))
        X_split = X_split[order]
        y_split = y_split[order]
        X_parts.append(X_split)
        y_parts.append(y_split)
        split_parts.append(np.full(len(y_split), split_name))
        print(f"{split_name}: X={X_split.shape}, y={y_split.shape}")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    split = np.concatenate(split_parts, axis=0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=X,
        y=y,
        split=split,
        labels=np.array(target_labels),
        sample_rate=16000,
        source="speech_commands_v0.02_tarballs",
    )
    print(f"saved: {output}")
    print(f"X: {X.shape}, y: {y.shape}, labels: {target_labels}")


if __name__ == "__main__":
    main()
