from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn, tensor
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram
from torchvision.models import ResNet18_Weights, resnet18

BATCH_SIZE = 16
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
EPOCHS = 20
EPS = 1e-6
LEARING_RATE = 1e-4
OUTPUT_MODEL = "fan_cnn.pth"
SEED = 42
THRESHOLD = 0.1


def build_mel(sample_rate: int) -> MelSpectrogram:
    return MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=128
    )


def log_mel(
    waveform: Tensor, sample_rate: int, mel: MelSpectrogram | None = None
) -> Tensor:
    mel = mel or build_mel(sample_rate)
    return torch.log(mel(waveform) + EPS)


class Fan(Dataset):
    def __init__(self, data_folders: list[str]):
        self.paths = sorted(
            path for folder in data_folders for path in Path(folder).glob("*.wav")
        )

        # Assume all audio have the same sample rate
        _, sample_rate = torchaudio.load(self.paths[0])
        self.sample_rate = sample_rate
        self.mel = build_mel(sample_rate)
        self.labels = [int("anomaly" in p.name) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        waveform, _ = torchaudio.load(self.paths[i])
        log_spectrogramme = log_mel(waveform, self.sample_rate, self.mel)
        label = self.labels[i]
        return log_spectrogramme, label


def build_model():
    # C.N.N.: leverages spatial invariance
    # translation invariance: earliest layers should respond similarly to the same patch
    # locality principle: earliest layers should focus on local regions
    # deeper layers should represent larger and more complex aspects of an image

    # https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # RGB (3 ch) → spectrogram (1 ch)
    # print(model.conv1)
    # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1.in_channels = 1
    model.conv1.weight = nn.Parameter(
        model.conv1.weight.data.mean(dim=1, keepdim=True) * 3
    )

    # Dropout can be applied
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.3),
    #     nn.Linear(model.fc.in_features, 1)
    # )
    # 1000 classes → 1 anomaly score
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def train(data_folders: list[str]):
    fan = Fan(data_folders)
    labels = fan.labels

    training, validation = (
        DataLoader(
            torch.utils.data.Subset(fan, indices),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
        )
        for shuffle, indices in zip(
            [True, False],
            train_test_split(
                range(len(fan)), test_size=0.2, stratify=labels, random_state=SEED
            ),  # test : training = 1 : 4
        )
    )

    model = build_model().to(DEVICE)
    n_anomaly = sum(labels)

    # Binary cross-entropy loss
    pos_weight = tensor([(len(labels) - n_anomaly) / n_anomaly]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = Adam(model.parameters(), lr=LEARING_RATE)

    # Decays the learning rate by gamma
    scheduler = StepLR(optimiser, step_size=10, gamma=0.3)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for log_spectrogrammes, labels in training:
            optimiser.zero_grad()
            prediction = model(log_spectrogrammes.to(DEVICE)).squeeze(1)
            loss = criterion(prediction, labels.float().to(DEVICE))
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        probabilities, targets = [], []
        with torch.no_grad():
            for log_spectrogrammes, labels in validation:
                probabilities.extend(
                    torch.sigmoid(model(log_spectrogrammes.to(DEVICE)).squeeze(1))
                    .cpu()
                    .tolist()
                )
                targets.extend(labels.tolist())

        auc = roc_auc_score(targets, probabilities)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} loss={train_loss / len(training):.4f} auc={auc:.4f}"
        )

    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Saved to '{OUTPUT_MODEL}'")


def predict(file: str):
    waveform, sample_rate = torchaudio.load(file)
    log_spectrogramme = log_mel(waveform, sample_rate)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(OUTPUT_MODEL, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        probablity = torch.sigmoid(
            model(log_spectrogramme.unsqueeze(0).to(DEVICE)).squeeze(1)
        ).item()

    return probablity


if __name__ == "__main__":
    train(["data/dev", "data/dev/abundant"])
    predict("data/dev/model_00_anomaly_00000108.wav")
    predict("data/dev/model_06_normal_00000030.wav")
