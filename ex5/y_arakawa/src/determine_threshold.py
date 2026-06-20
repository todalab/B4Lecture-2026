"""正常音データと非常音データに対して学習したモデルに通して、誤差から最適なしきい値を決定する."""

from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from torch import nn

from dataloaders.dataloader import MelSpectrogramDataset, create_dataloader
from make_plot import drawplots
from models.autoencoder import Autoencoder


def determine_threshold(config_path: str, train_file_list: str, eval_file_list: str, data_dir: str):
    """正常音データと非常音データに対して学習したモデルに通して、誤差から最適なしきい値を決定する."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 学習のときに使った設定を読みこむ
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log_dir = cfg["hydra"]["sweep"]["dir"]

    # 最適な隠れ層のチャンネル数の数を読みこむ
    with open(Path(log_dir) / "optimization_results.yaml", "r", encoding="utf-8") as f:
        # yaml形式は辞書形式で扱うことができる
        data = yaml.safe_load(f)

    hidden_channels1 = 32  # data["best_params"]["model.hidden_channels"]
    hidden_channels2 = 16  # data["best_params"]["model.hidden_channels"]
    learning_rate = data["best_params"]["train.learning_rate"]

    train_list_path = Path(train_file_list)
    eval_list_path = Path(eval_file_list)
    data_dir_path = Path(data_dir)
    db_min, db_max = MelSpectrogramDataset.compute_db_min_max(
        train_list_path,
        data_dir_path,
        sample_rate=cfg["dataset"]["sample_rate"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        n_mels=cfg["dataset"]["n_mels"],
        target_frames=cfg["dataset"]["target_frames"],
        device=torch.device("cpu"),
    )

    train_loader = create_dataloader(
        train_list_path,
        data_dir_path,
        batch_size=1,
        shuffle=True,
        sample_rate=cfg["dataset"]["sample_rate"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        n_mels=cfg["dataset"]["n_mels"],
        target_frames=cfg["dataset"]["target_frames"],
        seed=int(cfg["train"]["seed"]),
        device=device,
        db_min=db_min,
        db_max=db_max,
    )
    val_loader = create_dataloader(
        eval_list_path,
        data_dir_path,
        batch_size=1,  # 評価は1サンプルずつ行う
        shuffle=True,
        sample_rate=cfg["dataset"]["sample_rate"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        n_mels=cfg["dataset"]["n_mels"],
        target_frames=cfg["dataset"]["target_frames"],
        seed=int(cfg["train"]["seed"]),
        device=device,
        db_min=db_min,
        db_max=db_max,
    )

    model = Autoencoder(
        in_channels=1,
        hidden_channels1=hidden_channels1,
        hidden_channels2=hidden_channels2,
        latent_channels=cfg["model"]["latent_channels"],
        n_mels=cfg["dataset"]["n_mels"],
        target_frames=cfg["dataset"]["target_frames"],
    ).to(device)
    model.load_state_dict(
        torch.load(Path(log_dir) / Path(f"lr{round(learning_rate, 4)}") / Path("ckpts") / Path("model_epoch_0009.pt"))
    )
    loss_fn = nn.MSELoss()

    # 正解ラベルを格納するリスト
    y_true = np.array([])
    # モデルの出力と正解ラベルの誤差を格納するリスト
    y_score = np.array([])
    # 誤差を求める（評価データで閾値を決める）
    model.eval()  # DropoutやBatchNormを評価モードにする
    with torch.no_grad():  # 勾配を計算しない
        for mels, label, _ in val_loader:  # 評価データを使う
            mels = mels.to(device)
            recon = model(mels)
            loss = loss_fn(recon, mels)
            y_score = np.append(y_score, loss.item())
            y_true = np.append(y_true, label.item())

    plots = drawplots(
        Score_data=y_score,
        Label_data=y_true,
        OK_score=[y_score[i] for i in range(len(y_score)) if y_true[i] == 0],
        NG_score=[y_score[i] for i in range(len(y_score)) if y_true[i] == 1],
    )
    plots.draw_histogram(threshold_opt=None)
    plt.savefig(Path(log_dir) / Path("data_distribution.png"))

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    # 2.1.AUC score
    roc_auc_value = roc_auc_score(y_true, y_score)
    # 3.f1 Value
    precision, recall, threshold_from_pr = precision_recall_curve(y_true, y_score)

    prec_for_thresh = precision[1:]
    rec_for_thresh = recall[1:]
    f1 = 2 * prec_for_thresh * rec_for_thresh / (prec_for_thresh + rec_for_thresh + 1e-12)
    idx_opt = int(np.argmax(f1))
    threshold_opt = threshold_from_pr[idx_opt]
    idx_opt_from_pr = np.where(threshold == threshold_opt)  # ROC Curve

    # 5 draw f1 score
    plots.draw_f1_score(threshold_from_pr=threshold_from_pr, f1=f1)
    plt.savefig(Path(log_dir) / Path("f1_score.png"))

    # 6.draw precision recall curve
    plots.draw_precision_recall(precision=precision, recall=recall)
    plt.savefig(Path(log_dir) / Path("precision_recall_curve.png"))

    # 7. Plot ROC curve
    plots.draw_ROC_curve(fpr=fpr, tpr=tpr, opt_idx=idx_opt_from_pr, roc_auc_value=roc_auc_value)
    plt.savefig(Path(log_dir) / Path("ROC_curve.png"))

    # 8. Draw Confusion Matrix
    plots.draw_confusion_matrix(Score_data=y_score, Label_data=y_true, threshold_opt=threshold_opt)
    plt.savefig(Path(log_dir) / Path("confusion_matrix.png"))

    # 09. Draw Histogram with Threshold
    plots.draw_histogram(threshold_opt=threshold_opt)
    plt.savefig(Path(log_dir) / Path("data_distribution_with_threshold.png"))

    plt.show()

    plt.close("all")


if __name__ == "__main__":
    determine_threshold(
        config_path="configs/config_optuna_for_comp.yaml",
        train_file_list="src/dataloaders/normal_and_anomaly_data_train.txt",
        eval_file_list="src/dataloaders/normal_and_anomaly_data_eval.txt",
        data_dir="../data/dev",
    )
