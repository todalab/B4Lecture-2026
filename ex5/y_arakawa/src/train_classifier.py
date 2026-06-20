"""AEの潜在特徴量に対してBCE損失でMLP分類器を学習する。"""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataloaders.dataloader import MelSpectrogramDataset, create_dataloader
from models.autoencoder import Autoencoder
from models.mlp_classifier import MLPClassifier
from utils.seed import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OmegaConf.register_new_resolver("round", round, replace=True)


def _resolve_list_path(path_like: str | Path) -> Path:
    """設定されたファイルリストのパスを解決し、拡張子が省略されていれば .txt を補う。"""
    raw_path = Path(str(path_like))
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [raw_path, Path.cwd() / raw_path, Path.cwd().parent / raw_path, repo_root / raw_path]

    for candidate in candidates:
        resolved = candidate if candidate.suffix else candidate.with_suffix(".txt")
        if resolved.exists():
            return resolved.resolve()

    fallback = raw_path if raw_path.suffix else raw_path.with_suffix(".txt")
    return fallback.resolve()


def _resolve_data_dir_path(path_like: str | Path | None) -> Path | None:
    """設定されたデータディレクトリのパスを（指定されていれば）解決する。"""
    if path_like is None:
        return None
    raw_path = Path(str(path_like))
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [raw_path, Path.cwd() / raw_path, Path.cwd().parent / raw_path, repo_root / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / raw_path).resolve()


def _resolve_ckpt_path(path_like: str | Path) -> Path:
    """よくある配置場所を探索してチェックポイントのパスを解決する。"""
    raw_path = Path(str(path_like))
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [raw_path, Path.cwd() / raw_path, Path.cwd().parent / raw_path, repo_root / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"AE checkpoint not found: {raw_path}")


def _resolve_best_ae_ckpt_path() -> Path:
    """Optuna の最適化結果から AE チェックポイントのパスを動的に解決する。

    パス書式:
        logs/mel_ae_optuna_for_comp/lr{best_lr を小数第4位に四捨五入}/ckpts/model_epoch_{E:04d}.pt
        ただし E = ((epochs - 1) // epoch_per_ckpt) * epoch_per_ckpt
    """
    repo_root = Path(__file__).resolve().parent.parent  # = ex5/y_arakawa
    sweep_root = repo_root / "logs" / "mel_ae_optuna_for_comp"
    optuna_results_path = sweep_root / "optimization_results.yaml"
    optuna_cfg_path = repo_root / "configs" / "config_optuna_for_comp.yaml"

    results = OmegaConf.load(optuna_results_path)
    sweep_cfg = OmegaConf.load(optuna_cfg_path)

    best_lr = float(results.best_params["train.learning_rate"])
    # sweep の subdir 命名 `lr${round:${train.learning_rate}, 4}` と同じ整形を行う。
    lr_dir = f"lr{round(best_lr, 4)}"

    epochs = int(sweep_cfg.train.epochs)
    epoch_per_ckpt = int(sweep_cfg.train.epoch_per_ckpt)
    # epochs-1 以下で最も大きい epoch_per_ckpt の倍数のエポック。
    best_epoch = ((epochs - 1) // epoch_per_ckpt) * epoch_per_ckpt
    ckpt_name = f"model_epoch_{best_epoch:04d}.pt"

    ckpt_path = sweep_root / lr_dir / "ckpts" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Auto-resolved AE checkpoint not found: {ckpt_path}")
    return ckpt_path.resolve()


def _encode(autoencoder: Autoencoder, mels: torch.Tensor, freeze_encoder: bool) -> torch.Tensor:
    """AEのエンコーダを実行する。必要に応じて勾配追跡を無効化する。"""
    if freeze_encoder:
        with torch.no_grad():
            return autoencoder.encode(mels)
    return autoencoder.encode(mels)


def evaluate(
    autoencoder: Autoencoder,
    classifier: MLPClassifier,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    freeze_encoder: bool,
) -> tuple[float, float, float]:
    """データローダ全体に対する平均BCE損失・2値正解率・ROC-AUCを計算する。"""
    autoencoder.eval()
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for mels, labels, _ in loader:
            mels = mels.to(device)
            labels = labels.to(device).float()
            latent = _encode(autoencoder, mels, freeze_encoder=True)
            logits = classifier(latent)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * mels.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_count += mels.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    if total_count == 0:
        return 0.0, 0.0, 0.5
    probs_arr = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    if np.unique(labels_arr).size < 2:
        auc_val = float("nan")
    else:
        auc_val = float(roc_auc_score(labels_arr, probs_arr))
    return total_loss / total_count, total_correct / total_count, auc_val


def train(cfg: DictConfig) -> float:
    """MLP分類器の学習ループを実行し、最良の検証損失を返す。"""
    set_seed(int(cfg.train.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_list_path = _resolve_list_path(cfg.dataset_normal_and_anomaly.train_list)
    val_list_path = _resolve_list_path(cfg.dataset_normal_and_anomaly.eval_list)
    data_dir_path = _resolve_data_dir_path(cfg.dataset_normal_and_anomaly.get("data_dir", cfg.dataset.data_dir))

    db_min, db_max = MelSpectrogramDataset.compute_db_min_max(
        train_list_path,
        data_dir_path,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        n_mels=cfg.dataset.n_mels,
        target_frames=cfg.dataset.target_frames,
        device=torch.device("cpu"),
    )

    train_loader = create_dataloader(
        train_list_path,
        data_dir_path,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        n_mels=cfg.dataset.n_mels,
        target_frames=cfg.dataset.target_frames,
        seed=int(cfg.train.seed),
        device=device,
        db_min=db_min,
        db_max=db_max,
    )
    val_loader = create_dataloader(
        val_list_path,
        data_dir_path,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        n_mels=cfg.dataset.n_mels,
        target_frames=cfg.dataset.target_frames,
        seed=int(cfg.train.seed),
        device=device,
        db_min=db_min,
        db_max=db_max,
    )
    print("GPU available: ", torch.cuda.is_available())

    # 事前学習済みAEを構築して読み込む。
    variant = str(cfg.model.get("variant", "fc"))
    autoencoder = Autoencoder(
        in_channels=1,
        hidden_channels1=cfg.model.hidden_channels1,
        hidden_channels2=cfg.model.hidden_channels2,
        latent_channels=cfg.model.latent_channels,
        variant=variant,
        n_mels=cfg.dataset.n_mels,
        target_frames=cfg.dataset.target_frames,
    ).to(device)

    ae_ckpt_cfg = cfg.classifier.get("ae_ckpt", None)
    if ae_ckpt_cfg is None or str(ae_ckpt_cfg).strip().lower() in ("", "auto", "null", "none"):
        ae_ckpt_path = _resolve_best_ae_ckpt_path()
        logger.info(f"Auto-resolved AE checkpoint from Optuna results: {ae_ckpt_path}")
    else:
        ae_ckpt_path = _resolve_ckpt_path(ae_ckpt_cfg)
    state_dict = torch.load(ae_ckpt_path, map_location=device)
    autoencoder.load_state_dict(state_dict)
    logger.info(f"Loaded AE checkpoint from {ae_ckpt_path}")

    freeze_encoder = bool(cfg.classifier.get("freeze_encoder", True))
    if freeze_encoder:
        for p in autoencoder.parameters():
            p.requires_grad = False
        autoencoder.eval()

    # ダミー入力で順伝播し、潜在表現の次元数を取得する。
    with torch.no_grad():
        dummy = torch.zeros(1, 1, cfg.dataset.n_mels, cfg.dataset.target_frames, device=device)
        latent_shape = autoencoder.encode(dummy).shape
    input_dim = int(torch.tensor(latent_shape[1:]).prod().item())
    logger.info(f"Latent shape: {tuple(latent_shape)} -> MLP input_dim={input_dim}")

    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=list(cfg.classifier.get("hidden_dims", [64, 32])),
        dropout=float(cfg.classifier.get("dropout", 0.0)),
    ).to(device)

    trainable_params: list[torch.nn.Parameter] = list(classifier.parameters())
    if not freeze_encoder:
        trainable_params += list(autoencoder.parameters())
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=cfg.train.learning_rate,
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )

    # クラス不均衡の補正: BCEWithLogitsLoss に pos_weight = N_neg / N_pos を設定する。
    pos_weight_tensor: torch.Tensor | None = None
    if bool(cfg.classifier.get("use_pos_weight", True)):
        n_pos = 0
        n_neg = 0
        for _, labels, _ in train_loader:
            n_pos += int((labels == 1).sum().item())
            n_neg += int((labels == 0).sum().item())
        if n_pos > 0 and n_neg > 0:
            pos_weight_value = n_neg / n_pos
            pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
            logger.info(f"Train class counts: normal={n_neg}, anomaly={n_pos} -> pos_weight={pos_weight_value:.4f}")
        else:
            logger.warning(f"Skipping pos_weight (normal={n_neg}, anomaly={n_pos}); one class is empty.")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    log_dir = Path(str(cfg.train.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(str(cfg.train.ckpt_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # TensorBoard上で train/val の曲線を同一チャート内に異なる色で表示するため、
    # サブディレクトリごとに別のwriterを用意する（サブディレクトリ単位で1つのrunとなる）。
    writer_train = SummaryWriter(log_dir=str(log_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(log_dir / "val"))

    best_val_loss = float("inf")
    best_val_auc = -float("inf")

    try:
        for epoch in tqdm(range(cfg.train.epochs)):
            if not freeze_encoder:
                autoencoder.train()
            classifier.train()
            running_loss = 0.0
            running_correct = 0
            running_count = 0

            for mels, labels, _ in train_loader:
                mels = mels.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()
                latent = _encode(autoencoder, mels, freeze_encoder=freeze_encoder)
                logits = classifier(latent)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * mels.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                running_correct += (preds == labels.long()).sum().item()
                running_count += mels.size(0)

            avg_train_loss = running_loss / max(running_count, 1)
            train_acc = running_correct / max(running_count, 1)
            writer_train.add_scalar("loss", avg_train_loss, epoch)
            writer_train.add_scalar("acc", train_acc, epoch)

            if epoch % cfg.train.epoch_per_val == 0:
                val_loss, val_acc, val_auc = evaluate(
                    autoencoder, classifier, val_loader, device, loss_fn, freeze_encoder
                )
                best_val_loss = min(best_val_loss, val_loss)
                writer_val.add_scalar("loss", val_loss, epoch)
                writer_val.add_scalar("acc", val_acc, epoch)
                writer_val.add_scalar("auc", val_auc, epoch)
                logger.info(
                    f"Epoch {epoch:03d} | train_loss {avg_train_loss:.4f} | "
                    f"train_acc {train_acc:.4f} | val_loss {val_loss:.4f} | "
                    f"val_acc {val_acc:.4f} | val_auc {val_auc:.4f}"
                )
                if not np.isnan(val_auc) and val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_ckpt_path = ckpt_dir / "classifier_best.pt"
                    torch.save(
                        {
                            "classifier": classifier.state_dict(),
                            "autoencoder": autoencoder.state_dict(),
                            "freeze_encoder": freeze_encoder,
                            "epoch": epoch,
                            "val_auc": val_auc,
                        },
                        best_ckpt_path,
                    )
                    logger.info(f"Saved best checkpoint (val_auc={val_auc:.4f}) -> {best_ckpt_path}")

            if epoch % cfg.train.epoch_per_ckpt == 0:
                ckpt_path = ckpt_dir / f"classifier_epoch_{epoch:04d}.pt"
                torch.save(
                    {
                        "classifier": classifier.state_dict(),
                        "autoencoder": autoencoder.state_dict(),
                        "freeze_encoder": freeze_encoder,
                    },
                    ckpt_path,
                )
    finally:
        writer_train.close()
        writer_val.close()

    return best_val_loss


@hydra.main(version_base=None, config_path="../configs", config_name="config_classifier")
def main(cfg: DictConfig) -> float:
    return train(cfg)


if __name__ == "__main__":
    main()
