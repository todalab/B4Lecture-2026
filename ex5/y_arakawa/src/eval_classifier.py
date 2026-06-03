"""AEエンコーダ + MLPClassifier をテストセットで評価し、ROC曲線を描画する。"""

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, roc_curve

from dataloaders.dataloader import MelSpectrogramDataset, create_dataloader
from models.autoencoder import Autoencoder
from models.mlp_classifier import MLPClassifier
from utils.seed import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OmegaConf.register_new_resolver("round", round, replace=True)


def _resolve_list_path(path_like: str | Path) -> Path:
    """設定されたファイルリストのパスを解決し、拡張子が省略されている場合は .txt を付与する。"""
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
    """指定されている場合、設定されたデータディレクトリのパスを解決する。"""
    if path_like is None:
        return None
    raw_path = Path(str(path_like))
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [raw_path, Path.cwd() / raw_path, Path.cwd().parent / raw_path, repo_root / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / raw_path).resolve()


def _resolve_existing_path(path_like: str | Path) -> Path:
    """存在することが前提のパスを、よくある場所を探索して解決する。"""
    raw_path = Path(str(path_like))
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [raw_path, Path.cwd() / raw_path, Path.cwd().parent / raw_path, repo_root / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Path not found: {raw_path}")


def _load_ae_state_dict(ckpt: dict | torch.Tensor, fallback_ae_ckpt: str | Path | None) -> dict:
    """統合チェックポイントから AE の state_dict を返す。無ければフォールバックパスから読み込む。"""
    if isinstance(ckpt, dict) and "autoencoder" in ckpt:
        return ckpt["autoencoder"]
    if fallback_ae_ckpt is None:
        raise ValueError(
            "チェックポイントに 'autoencoder' キーが含まれていません。"
            "AE を別途読み込むには、設定で classifier.ae_ckpt を指定してください。"
        )
    ae_path = _resolve_existing_path(fallback_ae_ckpt)
    logger.info(f"AE の state_dict を {ae_path} から読み込みます")
    return torch.load(ae_path, map_location="cpu")


def _load_classifier_state_dict(ckpt: dict | torch.Tensor) -> dict:
    """統合チェックポイントまたは生の state_dict から分類器の state_dict を返す。"""
    if isinstance(ckpt, dict) and "classifier" in ckpt:
        return ckpt["classifier"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("対応していない分類器チェックポイント形式です。")


@torch.no_grad()
def _collect_scores(
    autoencoder: Autoencoder,
    classifier: MLPClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """ローダ全体に対してエンコーダ + 分類器を実行し、確率・ラベル・機種IDを返す。"""
    autoencoder.eval()
    classifier.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_model_ids: list[str] = []
    for mels, labels, model_ids in loader:
        mels = mels.to(device)
        latent = autoencoder.encode(mels)
        logits = classifier(latent)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
        all_model_ids.extend(list(model_ids))
    return np.concatenate(all_probs), np.concatenate(all_labels), all_model_ids


def _select_threshold_by_f1_hmean(
    probs: np.ndarray,
    labels: np.ndarray,
    model_ids: list[str],
    grid: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[float, float, dict[str, float]]:
    """しきい値を走査し、機種ごとの F1 の調和平均が最大となるしきい値を選ぶ。

    戻り値は (最良しきい値, 最良調和平均, 最良時の機種別F1) のタプル。
    全 model_id に同一のしきい値を適用する。あるしきい値で陽性または予測陽性が無い
    機種は F1=0 として寄与する（調和平均が定義できるよう ``eps`` でフロアする）。
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 991)
    model_ids_arr = np.asarray(model_ids)
    unique_models = np.unique(model_ids_arr)
    if unique_models.size == 0:
        return 0.5, 0.0, {}

    # 内側ループでのマスク計算を避けるため、機種ごとに probs/labels を事前分割する。
    per_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mid in unique_models:
        mask = model_ids_arr == mid
        per_model[str(mid)] = (probs[mask], labels[mask].astype(int))

    best_t = float(grid[0])
    best_score = -1.0
    best_per_model: dict[str, float] = {}
    for t in grid:
        f1s: list[float] = []
        per_model_f1: dict[str, float] = {}
        for mid, (p, y) in per_model.items():
            pred = (p >= t).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            if tp == 0:
                f1 = 0.0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2.0 * precision * recall / (precision + recall)
            per_model_f1[mid] = f1
            f1s.append(max(f1, eps))
        # 機種ごとの F1 の調和平均。
        hmean = len(f1s) / float(np.sum(1.0 / np.asarray(f1s)))
        if hmean > best_score:
            best_score = float(hmean)
            best_t = float(t)
            best_per_model = per_model_f1
    return best_t, best_score, best_per_model


def _plot_roc(
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    title: str,
    threshold: float | None = None,
) -> float:
    """ROC を描画して画像を保存する。AUC を返し、しきい値が与えられた場合は動作点を打点する。"""
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="C0", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    if threshold is not None and thr.size > 0:
        # roc_curve はしきい値を降順で返すため、指定値に最も近いものを探す。
        op_idx = int(np.argmin(np.abs(thr - threshold)))
        ax.scatter(
            fpr[op_idx],
            tpr[op_idx],
            color="C3",
            s=70,
            zorder=5,
            label=f"Threshold={threshold:.3f}\n(FPR={fpr[op_idx]:.3f}, TPR={tpr[op_idx]:.3f})",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return roc_auc


def _plot_prob_histogram(
    labels: np.ndarray, probs: np.ndarray, threshold: float, output_path: Path, title: str
) -> None:
    """MLP 出力確率のクラスごとのヒストグラムを描画する。"""
    normal_probs = probs[labels == 0]
    anomaly_probs = probs[labels == 1]
    bins = np.linspace(0.0, 1.0, 31)
    fig, ax = plt.subplots(figsize=(7, 5))
    if normal_probs.size > 0:
        ax.hist(
            normal_probs,
            bins=bins,
            alpha=0.6,
            color="C0",
            label=f"Normal (n={normal_probs.size})",
            edgecolor="white",
        )
    if anomaly_probs.size > 0:
        ax.hist(
            anomaly_probs,
            bins=bins,
            alpha=0.6,
            color="C3",
            label=f"Anomaly (n={anomaly_probs.size})",
            edgecolor="white",
        )
    ax.axvline(threshold, color="k", linestyle="--", lw=1, label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("MLP output probability (anomaly)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_prob_boxplot(labels: np.ndarray, probs: np.ndarray, output_path: Path, title: str) -> None:
    """クラスごとの確率の箱ひげ図 + ストリッププロットを描画する。"""
    normal_probs = probs[labels == 0]
    anomaly_probs = probs[labels == 1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [normal_probs, anomaly_probs],
        labels=[f"Normal (n={normal_probs.size})", f"Anomaly (n={anomaly_probs.size})"],
        widths=0.5,
        showfliers=False,
    )
    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate([(normal_probs, "C0"), (anomaly_probs, "C3")], start=1):
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(np.full_like(vals, i) + jitter, vals, alpha=0.5, s=18, color=color)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("MLP output probability (anomaly)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, output_path: Path, title: str) -> None:
    """2x2 の混同行列を描画する。"""
    cm = np.zeros((2, 2), dtype=int)
    for true_v, pred_v in zip(labels.astype(int), preds.astype(int), strict=False):
        cm[true_v, pred_v] += 1
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Normal", "Pred Anomaly"])
    ax.set_yticks([0, 1], labels=["True Normal", "True Anomaly"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
            )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(cfg: DictConfig) -> float:
    """テストセットで推論を行い、ROC 曲線と CSV を出力する。"""
    set_seed(int(cfg.train.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU 利用可否:", torch.cuda.is_available())

    eval_cfg = cfg.eval
    test_list_key = str(eval_cfg.get("test_list_key", "test_list"))
    test_list_path = _resolve_list_path(cfg.dataset_normal_and_anomaly[test_list_key])
    data_dir_path = _resolve_data_dir_path(cfg.dataset_normal_and_anomaly.get("data_dir", cfg.dataset.data_dir))

    # 学習時の統計量に合わせるため、db_min/db_max は *学習用* の normal+anomaly リストから算出する。
    train_list_path = _resolve_list_path(cfg.dataset_normal_and_anomaly.train_list)
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

    test_loader = create_dataloader(
        test_list_path,
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
    logger.info(f"テストリスト: {test_list_path} (n={len(test_loader.dataset)})")

    # モデルを構築する。
    variant = str(cfg.model.get("variant", "fc"))
    autoencoder = Autoencoder(
        in_channels=1,
        hidden_channels1=cfg.model.hidden_channels1,
        hidden_channels2=cfg.model.hidden_channels2,
        latent_channels=cfg.model.latent_channels,
        variant=variant,
    ).to(device)

    # 潜在次元をダミー入力で確認する。
    with torch.no_grad():
        dummy = torch.zeros(1, 1, cfg.dataset.n_mels, cfg.dataset.target_frames, device=device)
        latent_shape = autoencoder.encode(dummy).shape
    input_dim = int(torch.tensor(latent_shape[1:]).prod().item())
    logger.info(f"潜在表現の形状: {tuple(latent_shape)} -> MLP input_dim={input_dim}")

    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=list(cfg.classifier.get("hidden_dims", [64, 32])),
        dropout=float(cfg.classifier.get("dropout", 0.0)),
    ).to(device)

    # 分類器のチェックポイントを読み込む（既定では AE の state_dict も含まれている）。
    clf_ckpt_path = _resolve_existing_path(eval_cfg.classifier_ckpt)
    logger.info(f"分類器チェックポイントを {clf_ckpt_path} から読み込みます")
    ckpt = torch.load(clf_ckpt_path, map_location=device)
    autoencoder.load_state_dict(_load_ae_state_dict(ckpt, cfg.classifier.get("ae_ckpt")))
    classifier.load_state_dict(_load_classifier_state_dict(ckpt))

    # テストセットで推論を実行する。
    probs, labels, model_ids = _collect_scores(autoencoder, classifier, test_loader, device)

    # しきい値の選択。
    threshold_mode = str(eval_cfg.get("threshold_mode", "fixed"))
    if threshold_mode in ("youden_val", "f1_hmean_val"):
        val_list_path = _resolve_list_path(cfg.dataset_normal_and_anomaly.eval_list)
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
        val_probs, val_labels, val_model_ids = _collect_scores(autoencoder, classifier, val_loader, device)
        if np.unique(val_labels).size < 2:
            logger.warning("バリデーション集合のクラスが1つしかありません。固定しきい値にフォールバックします。")
            threshold = float(eval_cfg.get("threshold", 0.5))
        elif threshold_mode == "youden_val":
            v_fpr, v_tpr, v_thr = roc_curve(val_labels, val_probs)
            j_scores = v_tpr - v_fpr
            best_idx = int(np.argmax(j_scores))
            threshold = float(v_thr[best_idx])
            logger.info(f"バリデーションの Youden's J でしきい値を選択: {threshold:.4f} (J={j_scores[best_idx]:.4f})")
        else:  # f1_hmean_val
            grid_n = int(eval_cfg.get("f1_grid_size", 991))
            grid = np.linspace(0.01, 0.99, grid_n)
            threshold, best_hmean, per_model_f1 = _select_threshold_by_f1_hmean(
                val_probs, val_labels, val_model_ids, grid=grid
            )
            logger.info(f"バリデーションの F1 調和平均でしきい値を選択: {threshold:.4f} (H-mean F1={best_hmean:.4f})")
            for mid, f1 in sorted(per_model_f1.items()):
                logger.info(f"  val F1[{mid}] @ {threshold:.4f} = {f1:.4f}")
    else:
        threshold = float(eval_cfg.get("threshold", 0.5))

    # 結果を保存する。
    output_dir = Path(str(eval_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "test_predictions.csv"
    pd.DataFrame(
        {
            "model_id": model_ids,
            "label": labels.astype(int),
            "prob_anomaly": probs,
            "pred": (probs >= threshold).astype(int),
        }
    ).to_csv(csv_path, index=False)
    logger.info(f"予測結果を保存: {csv_path}")

    roc_path = output_dir / "roc_curve.png"
    roc_auc = _plot_roc(labels, probs, roc_path, title="Encoder + MLP (test)", threshold=threshold)
    logger.info(f"ROC 曲線を保存: {roc_path}")

    # 選択したしきい値でのクラス別精度を計算する。
    preds = (probs >= threshold).astype(int)

    hist_path = output_dir / "prob_histogram.png"
    _plot_prob_histogram(labels, probs, threshold, hist_path, title="MLP output distribution (test)")
    logger.info(f"確率ヒストグラムを保存: {hist_path}")

    box_path = output_dir / "prob_boxplot.png"
    _plot_prob_boxplot(labels, probs, box_path, title="MLP output per class (test)")
    logger.info(f"確率の箱ひげ図を保存: {box_path}")

    cm_path = output_dir / "confusion_matrix.png"
    _plot_confusion_matrix(labels, preds, cm_path, title=f"Confusion matrix @ threshold={threshold:.3f}")
    logger.info(f"混同行列を保存: {cm_path}")
    acc = float((preds == labels.astype(int)).mean()) if labels.size > 0 else 0.0
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    tpr = float(((preds == 1) & (labels == 1)).sum() / max(n_pos, 1))
    tnr = float(((preds == 0) & (labels == 0)).sum() / max(n_neg, 1))
    # 機種ごとの F1 とその調和平均（コンペの最終評価指標）。
    model_ids_arr = np.asarray(model_ids)
    per_model_f1: dict[str, float] = {}
    for mid in sorted(np.unique(model_ids_arr).tolist()):
        mask = model_ids_arr == mid
        y = labels[mask].astype(int)
        p = preds[mask].astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2.0 * precision * recall / (precision + recall)
        per_model_f1[str(mid)] = f1

    eps = 1e-12
    f1_vals = np.asarray([max(v, eps) for v in per_model_f1.values()])
    f1_hmean = float(len(f1_vals) / np.sum(1.0 / f1_vals)) if f1_vals.size > 0 else 0.0

    print("\n=== テスト結果 ===")
    print(f"正常数 = {n_neg}, 異常数 = {n_pos}")
    print(f"ROC-AUC              : {roc_auc:.4f}")
    print(f"Accuracy@{threshold:.2f}        : {acc:.4f}")
    print(f"TPR (異常の再現率)   : {tpr:.4f}")
    print(f"TNR (特異度)         : {tnr:.4f}")
    print(f"\n--- 機種別 F1 @ threshold={threshold:.4f} ---")
    for mid, f1 in per_model_f1.items():
        print(f"  F1[{mid}] = {f1:.4f}")
    print(f"F1 の調和平均: {f1_hmean:.4f}")

    return roc_auc


@hydra.main(version_base=None, config_path="../configs", config_name="config_classifier")
def main(cfg: DictConfig) -> float:
    return evaluate(cfg)


if __name__ == "__main__":
    main()
