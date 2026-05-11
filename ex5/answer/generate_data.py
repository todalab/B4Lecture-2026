import argparse
from pathlib import Path

import numpy as np


def make_envelope(n_samples, rng):
    attack = max(1, int(rng.uniform(0.02, 0.12) * n_samples))
    release = max(1, int(rng.uniform(0.08, 0.28) * n_samples))
    sustain = n_samples - attack - release
    if sustain < 0:
        sustain = 0
    sustain_level = rng.uniform(0.55, 1.0)
    env = np.concatenate(
        [
            np.linspace(0.0, 1.0, attack),
            np.full(sustain, sustain_level),
            np.linspace(sustain_level, 0.0, release),
        ]
    )
    return env[:n_samples]


TIMBRE_PROTOTYPES = np.array(
    [
        [1.00, 0.18, 0.10, 0.06, 0.03, 0.02, 0.01, 0.01],
        [1.00, 0.42, 0.30, 0.18, 0.10, 0.05, 0.03, 0.02],
        [1.00, 0.70, 0.24, 0.42, 0.16, 0.10, 0.06, 0.03],
        [1.00, 0.08, 0.62, 0.06, 0.34, 0.04, 0.16, 0.03],
    ],
    dtype=np.float64,
)


def oscillator(freq, t, rng):
    phase = rng.uniform(0.0, 2.0 * np.pi)
    vibrato_rate = rng.uniform(3.0, 7.0)
    vibrato_depth = rng.uniform(0.002, 0.018)
    phase_mod = vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * t)
    return np.sin(2.0 * np.pi * freq * t + phase + phase_mod)


def synth_timbre(label, base_freq, t, rng):
    prototype = TIMBRE_PROTOTYPES[label].copy()
    if rng.random() < 0.20:
        neighbor = (label + rng.choice([-1, 1])) % len(TIMBRE_PROTOTYPES)
        mix = rng.uniform(0.10, 0.25)
        prototype = (1.0 - mix) * prototype + mix * TIMBRE_PROTOTYPES[neighbor]

    amps = prototype * rng.lognormal(mean=0.0, sigma=0.16, size=len(prototype))
    wave = np.zeros_like(t)
    for harmonic, amp in enumerate(amps, start=1):
        detune = rng.normal(0.0, 0.0015)
        wave += amp * oscillator(base_freq * harmonic * (1.0 + detune), t, rng)
    return np.tanh(rng.uniform(0.7, 1.3) * wave)


def synth_chord(label, base_freq, t, rng):
    if label == 4:
        ratios = np.array([1.0, 1.25, 1.50])
    elif label == 5:
        ratios = np.array([1.0, 1.20, 1.50])
    else:
        raise ValueError(f"unknown chord label: {label}")

    if rng.random() < 0.20:
        ratios[1] += rng.normal(0.0, 0.010)

    wave = np.zeros_like(t)
    for ratio in ratios:
        amp = rng.uniform(0.35, 0.75)
        wave += amp * synth_timbre(rng.integers(0, len(TIMBRE_PROTOTYPES)), base_freq * ratio, t, rng)
    return wave / (np.max(np.abs(wave)) + 1e-8)


def synth_wave(label, t, rng):
    base_freq = np.exp(rng.uniform(np.log(170.0), np.log(760.0)))

    if label < len(TIMBRE_PROTOTYPES):
        wave = synth_timbre(label, base_freq, t, rng)
    else:
        wave = synth_chord(label, base_freq, t, rng)

    wave = wave * make_envelope(len(t), rng)
    wave = wave * rng.uniform(0.55, 1.0)
    wave = wave + rng.normal(0.0, rng.uniform(0.015, 0.045), size=len(t))

    if rng.random() < 0.20:
        delay = rng.integers(60, 220)
        echo = np.zeros_like(wave)
        echo[delay:] = wave[:-delay] * rng.uniform(0.08, 0.25)
        wave = wave + echo

    peak = np.max(np.abs(wave)) + 1e-8
    return (wave / peak).astype(np.float32)


def generate_dataset(samples_per_class, sample_rate, duration, seed):
    rng = np.random.default_rng(seed)
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    labels = np.array(
        ["soft_lead", "bright_lead", "reed_lead", "hollow_lead", "major_chord", "minor_chord"]
    )

    X = []
    y = []
    for label in range(len(labels)):
        for _ in range(samples_per_class):
            X.append(synth_wave(label, t, rng))
            y.append(label)

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    order = rng.permutation(len(y))
    return X[order], y[order], labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="answer/data/synthetic_music.npz")
    parser.add_argument("--samples-per-class", type=int, default=250)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y, labels = generate_dataset(
        samples_per_class=args.samples_per_class,
        sample_rate=args.sample_rate,
        duration=args.duration,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=X, y=y, labels=labels, sample_rate=args.sample_rate)
    print(f"saved: {output}")
    print(f"X: {X.shape}, y: {y.shape}, labels: {labels.tolist()}")


if __name__ == "__main__":
    main()
