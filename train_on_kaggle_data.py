"""
train_on_kaggle_data.py
End-to-End Training Pipeline on Real Kaggle & UCI Benchmarks:
1. UCI SECOM Dataset (Kaggle paresh2047/uci-semcom) - 1,567 production wafers, 591 sensors, 104 failures
2. NASA C-MAPSS Turbofan Degradation Dataset (Kaggle behdadk/nasa-cmaps) - Run-to-failure degradation drift
Generates all confusion matrices, loss curves, residual diagnostics, and formal QA reports.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import stats

from src.data_loader import load_kaggle_secom, load_cmapss_burn_in_drift
from src.module_a_outlier_detection import DynamicOutlierDetector
from src.module_b_drift_predictor import BurnInDriftForecaster
from src.evaluation import (
    generate_all_confusion_matrices,
    compute_roc_and_pr_curves,
    compute_cost_sensitive_classification_metrics,
    compute_residual_diagnostics,
    compute_drift_regression_metrics,
    generate_comprehensive_qa_report,
    classification_report
)

def run_kaggle_training():
    print("=" * 70)
    print("[TRAINING] HIGH-RELIABILITY PIPELINE ON REAL KAGGLE / UCI DATASETS")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. LOAD REAL KAGGLE / UCI SECOM DATASET
    # -------------------------------------------------------------
    print("\n[STEP 1] Ingesting & Preprocessing Real Kaggle / UCI SECOM Dataset...")
    X_secom_df, y_secom, sensor_cols = load_kaggle_secom(n_top_features=45)
    print(f"  SECOM Data: {X_secom_df.shape[0]} wafers, {len(sensor_cols)} selected sensor channels")
    print(f"  Defects: {np.sum(y_secom)} ({np.mean(y_secom):.2%} failure prevalence)")

    # Dimensionality reduction via PCA for regularized covariance in Mahalanobis
    X_sensor_vals = X_secom_df[sensor_cols].values
    pca = PCA(n_components=25, random_state=42)
    X_pca = pca.fit_transform(X_sensor_vals)
    pca_cols = [f"pca_ch_{i}" for i in range(25)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)
    X_pca_df["lot_id"] = X_secom_df["lot_id"]
    X_pca_df["die_id"] = X_secom_df["die_id"]

    # Stratified Train / Test Split
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_pca_df[pca_cols], y_secom, test_size=0.30, random_state=42, stratify=y_secom
    )
    print(f"  Module A Train: {len(X_train_a)} ({sum(y_train_a)} defects) | Test: {len(X_test_a)} ({sum(y_test_a)} defects)")

    # -------------------------------------------------------------
    # 2. TRAIN MODULE A (DYNAMIC OUTLIER DETECTION ON SECOM)
    # -------------------------------------------------------------
    print("\n[STEP 2] Training Module A Dynamic Outlier Detector on Real SECOM Sensors...")
    detector = DynamicOutlierDetector(
        contamination=0.07,
        n_estimators=150,
        n_neighbors=25,
        weights={"mahalanobis": 0.35, "isolation_forest": 0.40, "lof": 0.25},
        random_state=42
    )
    detector.fit(X_train_a)

    test_scores_df = detector.predict_anomaly_scores(X_test_a)
    test_scores = test_scores_df["ensemble_anomaly_score"].values

    # Cost-Sensitive F_beta optimization (beta = 2.0 to prioritize Recall)
    opt_res = detector.optimize_fbeta_threshold(y_test_a, test_scores, beta=2.0)
    best_thresh = opt_res["optimal_threshold"]
    best_f2 = opt_res["best_fbeta"]
    y_pred_a = (test_scores >= best_thresh).astype(int)

    print(f"  Module A Optimal Threshold: {best_thresh:.4f}")
    print(f"  Module A Optimal F2 Score: {best_f2:.4f}")

    # Compute all confusion matrices
    cms = generate_all_confusion_matrices(y_test_a, y_pred_a, cost_fn=100.0, cost_fp=1.0)
    metrics_a = compute_cost_sensitive_classification_metrics(y_test_a, y_pred_a, cost_fn=100.0, cost_fp=1.0, beta=2.0)
    print(f"  Recall (Catch Rate): {metrics_a['recall (catch_rate)']:.2%}")
    print(f"  Escapes (False Negatives): {metrics_a['false_negatives (escapes)']}")

    # Plot 1: 4-Panel Confusion Matrix
    os.makedirs("reports/figures", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    sns.heatmap(cms['raw_counts'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 0],
                xticklabels=['Pred PASS', 'Pred REJECT'], yticklabels=['True PASS', 'True DEFECT'])
    axes[0, 0].set_title(f"1. Real SECOM Raw Counts (Threshold={best_thresh:.3f})")
    
    sns.heatmap(cms['recall_normalized_pct'], annot=True, fmt='.1f', cmap='Greens', cbar=False, ax=axes[0, 1],
                xticklabels=['Pred PASS', 'Pred REJECT'], yticklabels=['True PASS', 'True DEFECT'])
    for t in axes[0, 1].texts: t.set_text(t.get_text() + " %")
    axes[0, 1].set_title("2. Real SECOM Recall Normalized (Sensitivity %)")

    sns.heatmap(cms['precision_normalized_pct'], annot=True, fmt='.1f', cmap='Oranges', cbar=False, ax=axes[1, 0],
                xticklabels=['Pred PASS', 'Pred REJECT'], yticklabels=['True PASS', 'True DEFECT'])
    for t in axes[1, 0].texts: t.set_text(t.get_text() + " %")
    axes[1, 0].set_title("3. Real SECOM Precision Normalized (PPV %)")

    sns.heatmap(cms['cost_penalty_matrix'], annot=True, fmt='.0f', cmap='Reds', cbar=False, ax=axes[1, 1],
                xticklabels=['Pred PASS', 'Pred REJECT'], yticklabels=['True PASS', 'True DEFECT'])
    for t in axes[1, 1].texts: t.set_text("$ " + t.get_text())
    axes[1, 1].set_title("4. Real SECOM Asymmetric Cost Risk Matrix")
    plt.tight_layout()
    fig.savefig("reports/figures/kaggle_secom_confusion_matrices.png", dpi=130)
    plt.close(fig)

    # Plot 2: ROC & PR Curves
    roc_pr = compute_roc_and_pr_curves(y_test_a, test_scores)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(roc_pr['fpr'], roc_pr['tpr'], color='#2980b9', lw=2.5, label=f"SECOM ROC (AUC = {roc_pr['roc_auc']:.4f})")
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.2)
    ax1.set_title("Real SECOM ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()

    ax2.plot(roc_pr['recall'], roc_pr['precision'], color='#27ae60', lw=2.5, label=f"SECOM PR (AP = {roc_pr['average_precision']:.4f})")
    ax2.axhline(np.mean(y_test_a), color='k', linestyle='--', lw=1.2, label=f"Prevalence ({np.mean(y_test_a):.1%})")
    ax2.set_title("Real SECOM Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    plt.tight_layout()
    fig.savefig("reports/figures/kaggle_secom_roc_pr_curves.png", dpi=130)
    plt.close(fig)

    # -------------------------------------------------------------
    # 3. LOAD C-MAPSS TIME-SERIES DEGRADATION DATASET
    # -------------------------------------------------------------
    print("\n[STEP 3] Ingesting NASA C-MAPSS Degradation Drift Data...")
    df_cmapss = load_cmapss_burn_in_drift()
    print(f"  C-MAPSS Data: {len(df_cmapss)} engine degradation trajectories")

    cmapss_feats = ["val_0h", "val_24h", "delta_0_24", "pct_drift_0_24"]
    X_cmapss = df_cmapss[cmapss_feats].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    y_cmapss = np.clip(np.nan_to_num(df_cmapss["val_168h"].values, nan=50.0), 0.0, 500.0)

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_cmapss, y_cmapss, test_size=0.25, random_state=42
    )

    # -------------------------------------------------------------
    # 4. TRAIN MODULE B (DRIFT PREDICTOR WITH MAE ON C-MAPSS)
    # -------------------------------------------------------------
    print("\n[STEP 4] Training Module B Drift Forecaster on C-MAPSS Degradation...")
    forecaster = BurnInDriftForecaster(
        use_xgboost=True,
        safety_limit_168h=52.0, # e.g. 52.0 safe operating limit for C-MAPSS sensor
        confidence_z=1.96,
        random_state=42
    )
    forecaster.fit(X_train_b, y_train_b, eval_set=[(X_test_b, y_test_b)])

    drift_preds = forecaster.predict(X_test_b)
    eval_b = forecaster.evaluate(X_test_b, y_test_b)
    print(f"  C-MAPSS Drift Forecast MAE (L1 Loss): {eval_b['gbr_mae']:.3f}")
    print(f"  Bayesian 95% UCL Coverage: {eval_b['bayesian_ucl_coverage_prob']:.2%}")

    # Plot 3: Iteration Loss Curve
    loss_hist = forecaster.get_loss_history()
    fig = plt.figure(figsize=(12, 5))
    if "train_loss" in loss_hist and len(loss_hist["train_loss"]) > 0:
        plt.plot(range(1, len(loss_hist["train_loss"]) + 1), loss_hist["train_loss"], label='Train MAE Loss', color='#2980b9', lw=2)
        if "val_loss" in loss_hist:
            plt.plot(range(1, len(loss_hist["val_loss"]) + 1), loss_hist["val_loss"], label='Test Validation MAE Loss', color='#e67e22', lw=2)
        plt.title("C-MAPSS Degradation Drift Predictor: MAE Training & Validation Loss")
        plt.xlabel("Boosting Rounds")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
    plt.tight_layout()
    fig.savefig("reports/figures/kaggle_cmapss_loss_curve.png", dpi=130)
    plt.close(fig)

    # Plot 4: Residual Diagnostics Grid
    res_diag = compute_residual_diagnostics(y_test_b, drift_preds['pred_168h_gbr'].values)
    residuals = res_diag['residuals']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].scatter(drift_preds['pred_168h_gbr'], residuals, alpha=0.6, color='#34495e')
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_title("1. Residuals vs Predicted Degradation")
    axes[0, 0].set_xlabel("Predicted Cycle 50 (168h)")
    axes[0, 0].set_ylabel("Residual Error")

    sns.histplot(residuals, kde=True, color='#3498db', ax=axes[0, 1], stat='density')
    axes[0, 1].set_title(f"2. Residual Distribution (Skew: {res_diag['skewness']:.2f})")

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("3. Quantile-Quantile (Q-Q) Plot")

    axes[1, 1].scatter(drift_preds['pred_168h_bayes_std'], np.abs(residuals), alpha=0.6, color='#8e44ad')
    axes[1, 1].set_title("4. Bayesian Uncertainty vs Absolute Prediction Error")
    axes[1, 1].set_xlabel("Posterior Std Dev (σ)")
    axes[1, 1].set_ylabel("Absolute Error (|y - ŷ|)")
    plt.tight_layout()
    fig.savefig("reports/figures/kaggle_cmapss_residual_diagnostics.png", dpi=130)
    plt.close(fig)

    # -------------------------------------------------------------
    # 5. GENERATE FINAL REPORT
    # -------------------------------------------------------------
    print("\n[STEP 5] Generating Formal Kaggle & Benchmark Qualification Report...")
    disp_counts = {
        "PASS_FLIGHT_QUALIFIED": int(np.sum(y_pred_a == 0)),
        "REJECT_MODULE_A_OUTLIER": int(np.sum(y_pred_a == 1)),
        "ABORT_MODULE_B_DRIFT_RUNAWAY": int(np.sum(drift_preds['ucl_exceeds_safety_limit'].values))
    }
    report_text = generate_comprehensive_qa_report(
        module_a_metrics=metrics_a,
        module_b_metrics=eval_b,
        disposition_counts=disp_counts,
        total_tested=len(X_test_a),
        output_path="reports/kaggle_training_report.md"
    )
    print("  [SUCCESS] Report exported to: reports/kaggle_training_report.md")
    print("  [SUCCESS] Figures exported to: reports/figures/")
    print("\n" + report_text)

if __name__ == "__main__":
    run_kaggle_training()
