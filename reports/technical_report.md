# Semiconductor Burn-In Screening and Drift Prediction Technical Report

## 1. Executive Summary
This project implements a dual-module quality screening pipeline for high-reliability semiconductor burn-in evaluation. The solution is designed for mission-critical environments where conventional specification limits are not sufficient because latent defects can appear normal in early-stage measurements yet drift rapidly toward failure during extended burn-in stress.

The system combines:
- Module A: contextual anomaly detection for early screening
- Module B: predictive drift forecasting for late-stage degradation
- Explainability and QA disposition logic to support auditability

The target operational objective is to minimize false negatives (escaped defects) while controlling yield loss and keeping the screening process transparent for engineering review.

---

## 2. Problem Statement
High-reliability semiconductor manufacturing is sensitive to lot-to-lot variation, process drift, and latent reliability defects that may not violate static pass/fail limits during early screening. In aerospace, satellite, defense, and high-availability electronics, these defects can lead to field failures with major operational costs.

The key challenge is that a component may:
- appear acceptable at 0h or 24h,
- remain within broad specification envelopes,
- yet exhibit accelerated degradation by 96h or 168h.

Therefore, a robust production system must look beyond fixed thresholds and model both contextual outlier behavior and time-dependent degradation trends.

---

## 3. System Architecture
The pipeline is organized into three technical layers.

### 3.1 Feature Engineering Layer
The feature layer constructs domain-informed signals that capture contextual variations and early drift acceleration:
- delta features: $\Delta_{0-24h}$ and percent drift
- lot-normalized Z-scores to compare each unit relative to its production lot
- stress interaction features representing operational acceleration conditions
- statistically normalized sensor behavior to reveal abnormal context-specific drift

These features let the model distinguish normal process variation from hazardous drift patterns.

### 3.2 Module A: Dynamic Outlier Detection
Module A performs early burn-in anomaly screening using a fused ensemble of detection methods:
- Mahalanobis distance with covariance regularization
- Isolation Forest
- Local Outlier Factor (LOF)

The ensemble score is thresholded using a cost-sensitive $F_\beta$ optimization strategy, with a bias toward reducing false negatives. This supports the mission-critical requirement to avoid escaped defects.

### 3.3 Module B: Drift Prediction
Module B estimates late-stage degradation using early-life measurements and drift velocity features. It uses:
- gradient boosting regression optimized on MAE (L1 loss)
- Bayesian ridge-based uncertainty estimation
- 95% upper confidence limit calculations for early abort assessments

This allows the system to identify devices whose projected failure risk exceeds the safety threshold before the end of the burn-in cycle.

### 3.4 Explainability & Quality Assurance
The explainability layer provides:
- SHAP-based feature attribution
- interpretable surrogate decision rules
- QA disposition summaries for engineering sign-off and auditability

This is essential for regulated or safety-critical manufacturing decisions.

---

## 4. Dataset and Operational Mapping
The implementation uses a dual-benchmark design:

| Module | Benchmark | Role |
| --- | --- | --- |
| Module A | UCI SECOM semiconductor defect dataset | Early contextual anomaly detection |
| Module B | NASA C-MAPSS degradation dataset | Late-stage drift forecasting |

This mapping allows the project to simulate the operational workflow of a burn-in and reliability screen while grounding the methodology in realistic manufacturing and engineering degradation patterns.

---

## 5. Feature Engineering Details
The feature engineering process focuses on physically meaningful indicators.

### 5.1 Early Drift Features
The project computes:
- $\Delta_{0-24h} = Value_{24h} - Value_{0h}$
- percentage drift relative to the initial value
- lot-specific normalized deviation for contextual evaluation

These features capture the fact that an abnormal drift velocity can be a stronger signal than the absolute magnitude alone.

### 5.2 Contextual Anomaly Signal
A major failure mode in semiconductor manufacturing is a die that appears normal relative to global specification but abnormal compared with its own lot or neighboring process conditions. The model explicitly addresses this by using lot-normalized Z-scores and interaction effects.

---

## 6. Methodology
### 6.1 Outlier Detection
The anomaly detector normalizes and combines the three model outputs into a single ensemble score. Threshold selection is optimized with cost-sensitive scoring that increases the penalty for false negatives, because escaped defects in aerospace and space hardware are operationally more dangerous than false alarms.

### 6.2 Drift Forecasting
The drift predictor estimates future degradation from early-stage measurements. It uses a loss function aligned with engineering needs: MAE minimizes absolute error in the physical quantity of interest. The Bayesian regression output provides uncertainty bounds and supports operational safety actions such as early abort when the upper confidence limit exceeds a safety threshold.

---

## 7. Results and Observations
Across the generated project benchmark and the real-data benchmark workflow, the model demonstrates the expected behavior for early-life screening:

- strong separation between normal and drift-accelerated behavior
- meaningful improvement in early reject decisions using contextual anomaly signals
- predictive degradation estimates that are usable for risk-triggered interventions
- calibrated uncertainty bounds supporting QA and engineering review

The existing project reports show the outcomes in both synthetic integrated screening runs and the benchmarked Kaggle/UCI workflow. The generated figures under the reports folder document the confusion matrices, ROC/PR curves, loss curves, and residual diagnostics.

---

## 8. Quantitative Highlights
The project includes the following summary metrics from the burn-in screening report:

| Metric | Value | Interpretation |
| --- | --- | --- |
| Total units screened | 1,250 | Screening volume |
| Flight qualified units | 1,123 (89.84%) | Cleared for operational use |
| Module A early rejects | 90 (7.20%) | Contextual anomalies caught early |
| Module B drift aborts | 37 (2.96%) | Prevents late-stage runaways |
| Escape count | 3 | Residual risk to monitor |

The more benchmark-heavy Kaggle-based run also shows a real-data training workflow with formal performance plots and QA reporting.

---

## 9. Deployment Guidance
This pipeline is best suited for deployment in a monitored manufacturing review flow with the following controls:
- early-stage anomaly screening at 24h
- predictive drift review based on early-life indicators
- engineering sign-off and SHAP-based audit trail for any reject or abort decision
- periodic re-calibration as process parameters or lot mixes change

For safety-critical applications, the system should be paired with human review for any condition near the decision threshold.

---

## 10. Conclusion
The implemented solution delivers a practical, auditable, and engineering-oriented burn-in screening framework. It moves beyond static pass/fail thresholds and combines contextual anomaly detection with predictive degradation modeling to support high-reliability manufacturing decisions.

The combination of explainability, performance reporting, and operational risk logic makes this project suitable for use as both an engineering analysis tool and a quality assurance decision support system.

---

## 11. Generated Deliverables
The project now includes:
- technical report: [reports/technical_report.md](reports/technical_report.md)
- presentation deck: [reports/semiconductor_burn_in_presentation.pptx](reports/semiconductor_burn_in_presentation.pptx)
- project summary reports: [reports/burn_in_screening_final_report.md](reports/burn_in_screening_final_report.md) and [reports/kaggle_training_report.md](reports/kaggle_training_report.md)
- supporting figures: [reports/figures](reports/figures)
