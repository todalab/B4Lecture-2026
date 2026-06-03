"""Train and run a ResNet18-based anomaly classifier C.N.N."""

from typing import Optional

import torch
import torchaudio
from fan import Fan, log_mel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn, tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

BATCH_SIZE = 16
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
EPOCHS = 20
LEARNING_RATE = 1e-4
OUTPUT_MODEL = "fan_cnn.pth"
SEED = 42
THRESHOLD = 0.1


def build_model() -> nn.Module:
    """Build a ResNet18 model adapted for spectrogramme classification."""
    # Tranfer learning (small dataset)

    # C.N.N.: leverages spatial invariance
    # translation invariance: earliest layers should respond similarly to the same patch
    # locality principle: earliest layers should focus on local regions
    # deeper layers should represent larger and more complex aspects of an image

    # https://arxiv.org/pdf/1512.03385
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    # print(model)
    # 18 layers

    # RGB (3 ch) → spectrogram (1 ch)
    # print(model.conv1)
    # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1.in_channels = 1
    model.conv1.weight = nn.Parameter(
        model.conv1.weight.data.mean(dim=1, keepdim=True) * 3
    )

    # Dropout:
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.3),
    #     nn.Linear(model.fc.in_features, 1)
    # )
    # ImageNet-1K 1000 object categories → 1 anomaly logit
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

    # For smaller dataset, there's a method to froze all parameters and
    # only update the parameters of the last linear layer
    # https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#exclusion-from-the-dag


def train(data_folders: list[str]):
    """Train the anomaly classifier and save the trained model."""
    fan = Fan(data_folders)
    labels = fan.labels

    training, validation = (
        DataLoader(
            torch.utils.data.Subset(fan, indices),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
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
    # sigmoid → binary cross entropy
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = Adam(model.parameters(), lr=LEARNING_RATE)
    # optimiser = SGD(model.parameters(), momentum=0.9)

    # Decays the learning rate by gamma
    scheduler = StepLR(optimiser, step_size=10, gamma=0.3)

    # https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for log_spectrogrammes, labels in training:
            optimiser.zero_grad()
            prediction = model(log_spectrogrammes.to(DEVICE)).squeeze(1)
            loss = criterion(prediction, labels.float().to(DEVICE))
            loss.backward()
            optimiser.step()  # gradient descent
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


def load_model() -> nn.Module:
    """Load the saved model and switch it to evaluation mode."""
    model = build_model().to(DEVICE)
    model.load_state_dict(
        torch.load(OUTPUT_MODEL, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model


model: Optional[nn.Module] = None


def predict(file: str):
    """Return the anomaly probability for a single audio file."""
    global model
    if model is None:
        model = load_model()  # will call model.eval()

    log_spectrogramme = log_mel(*torchaudio.load(file))

    with torch.no_grad():
        probability = torch.sigmoid(
            model(log_spectrogramme.unsqueeze(0).to(DEVICE)).squeeze(1)
        ).item()

    return probability
