import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from Config_RETIN import PLOT
import os


output_dir_paper = (f"/Users/claudiomazzi/Documents/PhD/PhD_Project/"
                    f"RETIN_CLASSIFIER/FINAL_RUN/PLOT/PLOT_PAPER/")
os.makedirs(output_dir_paper, exist_ok=True)

class PlotManager:
    def __init__(self, pdf_filename):
        self.pdf = PdfPages(pdf_filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pdf.close()

    def plot_roc_curve(self, y_test, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(output_dir_paper, 'ROC_curve.png'), dpi=350, bbox_inches='tight')
        self.pdf.savefig()
        plt.close()


    def plot_precision_recall_curve(self, y_test, y_pred_proba):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.title('Precision-Recall curve')

        plt.savefig(os.path.join(output_dir_paper, 'PRAUC_curve.png'), dpi=350, bbox_inches='tight')
        self.pdf.savefig()
        plt.close()

    def plot_confusion_matrix(self, y_test, y_pred_labels):
        cm = confusion_matrix(y_test, y_pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        # plt.title('Confusion Matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        self.pdf.savefig()
        plt.close()

    def plot_histogram_of_predictions(self, y_pred_proba):
        plt.figure()
        sns.histplot(y_pred_proba, bins=50, kde=True, color='blue')
        # plt.title('Histogram of Predicted Probabilities')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        self.pdf.savefig()
        plt.close()

    def plot_lift_chart(self, y_test, y_pred_proba):
        # Sort probabilities and calculate the lift
        data = np.column_stack((y_test, y_pred_proba))
        data = data[data[:, 1].argsort()[::-1]]  # Sort by probability (descending)
        total_positives = np.sum(y_test)
        cumulative_positives = np.cumsum(data[:, 0])
        lift = cumulative_positives / total_positives
        plt.figure()
        plt.plot(np.arange(len(lift)) / len(lift), lift, marker='.')
        # plt.title('Lift Chart')
        plt.xlabel('Percentage of Sample')
        plt.ylabel('Lift')
        self.pdf.savefig()
        plt.close()

    def plot_f1_precision_recall_vs_threshold(self, y_test, y_pred_proba):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores = np.nan_to_num(f1_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute e visualize best threshold
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        plt.figure()
        plt.plot(thresholds, f1_scores[:-1], label='F1 score')
        plt.plot(thresholds, precision[:-1], label='Precision')
        plt.plot(thresholds, recall[:-1], label='Recall')
        plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.4f}')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        # plt.title('Precision, Recall, F1 Score vs Threshold')
        plt.legend()

        plt.savefig(os.path.join(output_dir_paper, 'PR_F1_Rec_threshold.png'), dpi=350, bbox_inches='tight')
        self.pdf.savefig()
        plt.close()

    def plot_prob_distribution_by_class(self, y_test, y_pred_proba):
        plt.figure()
        sns.histplot(y_pred_proba[y_test == 0], bins=50, color='blue', label='Class 0', kde=True, stat="density")
        sns.histplot(y_pred_proba[y_test == 1], bins=50, color='orange', label='Class 1', kde=True, stat="density")
        # plt.title('Predicted Probability Distributions by Class')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend()

        plt.savefig(os.path.join(output_dir_paper, 'Prob_dist.png'), dpi=350, bbox_inches='tight')
        self.pdf.savefig()
        plt.close()

    #################################################################
    # JRS Plot                                                      #
    #################################################################
    def plot_PR_and_threshold(self, y_test, y_pred_proba):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores = np.nan_to_num(f1_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Best threshold
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]

        # === MULTIPLOT ===
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # --------------------------
        # (A) Precision–Recall curve
        # --------------------------
        axs[0].plot(recall, precision, lw=2)
        axs[0].set_xlabel("Recall")
        axs[0].set_ylabel("Precision")

        # panel label A
        axs[0].text(0.02, 0.98, "(A)", transform=axs[0].transAxes,
                    fontsize=14, fontweight="bold", va="top")

        # --------------------------
        # (B) Precision/Recall/F1 vs Threshold
        # --------------------------
        axs[1].plot(thresholds, f1_scores[:-1], label="F1 score", lw=2)
        axs[1].plot(thresholds, precision[:-1], label="Precision", lw=2)
        axs[1].plot(thresholds, recall[:-1], label="Recall", lw=2)
        axs[1].axvline(best_threshold, color="red", linestyle="--",
                       label=f"Best threshold = {best_threshold:.4f}")

        axs[1].set_xlabel("Threshold")
        axs[1].set_ylabel("Score")
        axs[1].legend()

        # panel label B
        axs[1].text(0.02, 0.98, "(B)", transform=axs[1].transAxes,
                    fontsize=14, fontweight="bold", va="top")

        plt.tight_layout()

        # === SAVE as VECTOR PDF ===
        output_path = os.path.join(output_dir_paper, "Fig2_PR_PRAUC.pdf")
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        self.pdf.savefig(fig)
        plt.close(fig)

        print("Saved JRSS-compliant Fig. 2 to:", output_path)


# Function to call all plots
def generate_all_plots(y_test, y_pred_proba, y_pred_labels, jrs_plot=True):
    with PlotManager(PLOT) as plot_manager:
        plot_manager.plot_roc_curve(y_test, y_pred_proba)
        plot_manager.plot_precision_recall_curve(y_test, y_pred_proba)
        plot_manager.plot_confusion_matrix(y_test, y_pred_labels)
        plot_manager.plot_histogram_of_predictions(y_pred_proba)
        plot_manager.plot_lift_chart(y_test, y_pred_proba)
        plot_manager.plot_f1_precision_recall_vs_threshold(y_test, y_pred_proba)
        plot_manager.plot_prob_distribution_by_class(y_test, y_pred_proba)

        if jrs_plot:
            plot_manager.plot_PR_and_threshold(y_test, y_pred_proba)
