import argparse
import platform
import subprocess
import wave
from pathlib import Path

import numpy as np


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["labels"], int(data["sample_rate"])


def write_wav(path, wave_data, sample_rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    wave_data = wave_data / (np.max(np.abs(wave_data)) + 1e-8)
    pcm = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def play_file(path):
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["afplay", str(path)], check=False)
    elif system == "Linux":
        subprocess.run(["aplay", str(path)], check=False)
    elif system == "Windows":
        subprocess.run(["powershell", "-c", f'(New-Object Media.SoundPlayer "{path}").PlaySync();'])
    else:
        print(f"playback is not supported on this platform: {system}")


def export_examples(X, y, labels, sample_rate, output_dir, per_class, play):
    written = []
    for label_id, label_name in enumerate(labels):
        indices = np.where(y == label_id)[0][:per_class]
        for rank, idx in enumerate(indices):
            path = output_dir / f"{label_name}_{rank:02d}_idx{idx:04d}.wav"
            write_wav(path, X[idx], sample_rate)
            written.append(path)

    print(f"wrote {len(written)} wav files to {output_dir}")
    if play and written:
        print(f"playing: {written[0]}")
        play_file(written[0])


def export_one(X, y, labels, sample_rate, output_dir, index, play):
    label_name = labels[y[index]]
    path = output_dir / f"{label_name}_idx{index:04d}.wav"
    write_wav(path, X[index], sample_rate)
    print(f"wrote: {path}")
    if play:
        play_file(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="answer/data/synthetic_music.npz")
    parser.add_argument("--output-dir", default="answer/audio")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()

    X, y, labels, sample_rate = load_npz(args.input)
    output_dir = Path(args.output_dir)

    if args.index is None:
        export_examples(X, y, labels, sample_rate, output_dir, args.per_class, args.play)
    else:
        if args.index < 0 or args.index >= len(y):
            raise ValueError(f"--index must be between 0 and {len(y) - 1}")
        export_one(X, y, labels, sample_rate, output_dir, args.index, args.play)


if __name__ == "__main__":
    main()
