import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


class drawplots:
    def __init__(self, Score_data, Label_data, OK_score, NG_score):
        self.Score_data = Score_data
        self.Label_data = Label_data
        self.OK_score = OK_score
        self.NG_score = NG_score

    def draw_histogram(self, threshold_opt=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.hist(self.OK_score, bins=100, color="blue", alpha=0.3, density=False, label="OK")
        ax.hist(self.NG_score, bins=100, color="red", alpha=0.3, density=False, label="NG")

        if threshold_opt:
            ax.axvline(x=threshold_opt, color="green", linestyle="dashed", linewidth=2)
            ax.set_title(f"OK,NG Data Distribution, threshold: {round(threshold_opt, 2)}")
        else:
            pass

        ax.set_title("OK,NG Data Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("frequency")
        ax.grid()
        ax.legend()

        return fig

    def draw_ROC_curve(self, fpr, tpr, opt_idx, roc_auc_value):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(fpr, tpr, marker="o")
        ax.plot(fpr[opt_idx], tpr[opt_idx], marker="o", color="red")
        ax.plot(np.linspace(1, 0, len(fpr)), np.linspace(1, 0, len(fpr)), linestyle="--", color="gray")

        ax.set_title(f"ROC Curve: {round(roc_auc_value, 3)}")
        ax.set_xlabel("FPR: False positive rate")
        ax.set_ylabel("TPR: True positive rate")

        ax.grid()
        ax.legend()

    def draw_f1_score(self, threshold_from_pr, f1):

        # find the optimal
        idx_opt = np.argmax(f1)
        threshold_opt = threshold_from_pr[idx_opt]
        # draw

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(threshold_from_pr, f1)
        ax.axvline(x=threshold_opt, color="green", linestyle="dashed", linewidth=2)
        ax.plot(threshold_from_pr[idx_opt], f1[idx_opt], marker="o", color="red")
        ax.set_title(f"F1 Score: {round(threshold_opt, 3)}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.grid()

        return fig

    def draw_confusion_matrix(
        self,
        Score_data,
        Label_data,
        threshold_opt,
    ):

        # make a predicted label
        predicted_label = []
        for kk in range(Score_data.shape[0]):
            if Score_data[kk] >= threshold_opt:
                predicted_label.append(1)  # NG
            else:
                predicted_label.append(0)

        # label_class
        classes = ["OK", "NG"]
        cm = confusion_matrix(y_true=Label_data, y_pred=predicted_label)
        cm = cm / cm.astype(np.float64).sum(axis=1) * 100

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        cmap = plt.cm.Blues
        ax.imshow(cm, cmap=cmap)
        for m in range(cm.shape[1]):
            for n in range(cm.shape[0]):
                ax.text(x=n, y=m, s=round(cm[m, n], 2), va="center", ha="center", color="gray")
        ax.set_title(f"Confusion Matrix, Optimal Threshold : {round(threshold_opt, 2)}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        return fig

    def draw_precision_recall(self, precision, recall):
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(precision, recall)
        ax.set_title("Precision Recall Curve")
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.grid()

        return fig
