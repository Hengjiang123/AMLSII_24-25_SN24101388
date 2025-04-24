import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from A.datasets import LesionDataset, get_transforms
from A.model import EfficientNetClassifier

# from amls2final7.amls2folder.A.datasets import LesionDataset, get_transforms
# from amls2final7.amls2folder.A.model import EfficientNetClassifier

def evaluate_test(test_df, config):
    print("\nEvaluating on hold-out test set...")
    os.makedirs(config.images_dir, exist_ok=True)

    # if not using preprocessed data, define transform
    if not config.use_preprocessed:
        _, _, val_transform = get_transforms(config)
    else:
        val_transform = None

    test_dataset = LesionDataset(
        test_df,
        weak_transform=val_transform,
        strong_transform=val_transform, 
        use_meta=config.use_meta,
        use_preprocessed=config.use_preprocessed,
        tensor_dir=config.tensor_dir
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    meta_input_dim = len(test_dataset.meta_cols) if config.use_meta else 0

    all_probs = []
    all_labels = []

    for fold in range(config.num_folds if not config.test_mode else 1):
        model = EfficientNetClassifier(
            num_classes=config.num_classes,
            use_meta=config.use_meta,
            meta_input_dim=meta_input_dim,
            backbone_name="efficientnet-b4"
        )
        model_path = os.path.join(config.weights_dir, f"best_model_fold{fold}.pth")
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model = model.to(config.device)
        model.eval()

        fold_probs = []
        fold_labels = []

        with torch.no_grad():
            for batch in test_loader:
                if config.use_meta:
                    images, meta, targets = batch
                    images, meta, targets = images.to(config.device), meta.to(config.device), targets.to(config.device)
                    outputs = model(images, meta)
                else:
                    images, targets = batch
                    images = images.to(config.device)
                    targets = targets.to(config.device)
                    outputs = model(images)

                prob = torch.softmax(outputs, dim=1)[:,1]
                fold_probs.extend(prob.cpu().numpy())
                fold_labels.extend(targets.cpu().numpy())

        all_probs.append(fold_probs)
        all_labels = fold_labels

    avg_probs = np.mean(all_probs, axis=0)
    y_true = np.array(all_labels)
    preds = (avg_probs >= 0.5).astype(int)  # threshold of 0.5

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc_ = roc_auc_score(y_true, avg_probs)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test AUC: {auc_:.4f}")
    print(f"Test Specificity: {specificity:.4f}")

    test_df["prob"] = avg_probs
    test_df["pred"] = preds
    test_df.to_csv(os.path.join(config.images_dir, "test_predictions.csv"), index=False)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_,
        "specificity": specificity
    }
    pd.DataFrame([metrics]).to_csv(
        os.path.join(config.images_dir, "test_metrics.csv"), index=False
    )

    # Confusion Matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(config.images_dir, "confusion_matrix_test.png"))
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, avg_probs)
    test_roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC={test_roc_auc:.3f})')
    plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve (Test Set)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(config.images_dir, "roc_curve_test.png"))
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, avg_probs)
    ap_score = average_precision_score(y_true, avg_probs)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR (AP={ap_score:.3f})')
    plt.title('Precision-Recall (Test Set)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(config.images_dir, "pr_curve_test.png"))
    plt.close()

    # Bar chart for Precision/Recall/F1
    plt.figure()
    metric_names = ['Precision', 'Recall', 'F1']
    metric_vals = [prec, rec, f1]
    plt.bar(metric_names, metric_vals)
    plt.title('PRF Bar Chart (Test Set)')
    plt.ylim([0,1])
    for i, v in enumerate(metric_vals):
        plt.text(i-0.1, v+0.01, f"{v:.3f}", color='black', fontweight='bold')
    plt.savefig(os.path.join(config.images_dir, "prf_bar_chart_test.png"))
    plt.close()

    # Threshold analysis 0.1 step
    def compute_metrics(tg, pd):
        acc_ = accuracy_score(tg, pd)
        prec_ = precision_score(tg, pd, zero_division=0)
        rec_ = recall_score(tg, pd, zero_division=0)
        f1_ = f1_score(tg, pd, zero_division=0)
        cm_ = confusion_matrix(tg, pd)
        tn_, fp_, fn_, tp_ = cm_.ravel()
        spec_ = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
        return acc_, prec_, rec_, f1_, spec_

    thresholds_1 = np.arange(0, 1.01, 0.1)
    results_1 = []
    for thr in thresholds_1:
        y_pred_thr = (avg_probs >= thr).astype(int)
        a, p, r, f_, s = compute_metrics(y_true, y_pred_thr)
        results_1.append({
            "threshold": thr,
            "accuracy": a,
            "precision": p,
            "recall": r,
            "f1": f_,
            "specificity": s
        })
    df_res1 = pd.DataFrame(results_1)
    df_res1.to_csv(os.path.join(config.images_dir,"threshold_metrics_0p1.csv"), index=False)

    # instegraph bar chart
    x_ = np.arange(len(thresholds_1))
    width = 0.15
    plt.figure(figsize=(12,6))
    metrics_list = ["accuracy","precision","recall","f1","specificity"]
    for i, m in enumerate(metrics_list):
        offset = (i - 2) * width
        plt.bar(x_ + offset, df_res1[m], width, label=m)
    plt.xticks(x_, [f"{t:.1f}" for t in thresholds_1])
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Metrics at different thresholds (step=0.1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.images_dir,"threshold_bar_0p1.png"))
    plt.close()

    # Threshold analysis 0.01 step
    thresholds_2 = np.arange(0, 1.001, 0.01)
    results_2 = []
    for thr in thresholds_2:
        y_pred_thr = (avg_probs >= thr).astype(int)
        a, p, r, f_, s = compute_metrics(y_true, y_pred_thr)
        results_2.append({
            "threshold": thr,
            "accuracy": a,
            "precision": p,
            "recall": r,
            "f1": f_,
            "specificity": s
        })
    df_res2 = pd.DataFrame(results_2)
    df_res2.to_csv(os.path.join(config.images_dir,"threshold_metrics_0p01.csv"), index=False)

    # 5 metrics line chart
    plt.figure(figsize=(10,6))
    for m in metrics_list:
        plt.plot(df_res2["threshold"], df_res2[m], label=m)
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Metrics vs Threshold (step=0.01)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.images_dir,"threshold_line_0p01.png"))
    plt.close()

    best_thresholds = {}
    for m in metrics_list:
        idxmax = df_res2[m].idxmax()
        best_thr = df_res2.loc[idxmax, "threshold"]
        best_val = df_res2.loc[idxmax, m]
        best_thresholds[m] = (best_thr, best_val)

    # print and save
    print("\nBest threshold for each metric (step=0.01):")
    thr_rows = []
    for m in metrics_list:
        thr, val = best_thresholds[m]
        print(f"  {m}: best threshold = {thr:.2f}, value = {val:.4f}")
        thr_rows.append({"metric":m, "best_threshold":thr, "best_value":val})
    pd.DataFrame(thr_rows).to_csv(os.path.join(config.images_dir,"best_thresholds.csv"), index=False)
    print("Saved best_thresholds.csv with each metric's best threshold & value.")
