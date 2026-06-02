"""Helper to draw spectrogramme."""

from pathlib import Path

import torchaudio
from matplotlib.pyplot import close, imshow, savefig, title

from mel import log_mel

Path("fig").mkdir(parents=True, exist_ok=True)

files = [
    "data/dev/model_03_anomaly_00000162.wav",
    "data/dev/model_03_anomaly_00000290.wav",
    "data/dev/model_04_anomaly_00000324.wav",
]

for file in files:
    path = Path(file)
    waveform, sample_rate = torchaudio.load(file)

    log_spectrogramme = log_mel(waveform, sample_rate)
    img = log_spectrogramme.squeeze(0).numpy()

    title(file)
    imshow(img, origin="lower", aspect="auto")
    savefig("fig/" + path.name.replace("wav", "png"))
    close()
