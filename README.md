# High-Reliability Semiconductor Burn-In Anomaly Detection & Time-Series Drift Predictor

An enterprise-grade, dual-module machine learning pipeline designed for high-reliability semiconductor manufacturing and burn-in screening (tailored for aerospace, satellite, and mission-critical specifications).

---

## 📌 Executive Architecture & Problem Overview

In mission-critical semiconductor manufacturing (e.g. space applications, launch vehicles, satellite transponders), static upper/lower specification limits are **insufficient**. Normal manufacturing variations cause lot-to-lot baseline shifts, while dangerous **latent defects** may look normal at early testing ($0h$) or within generic specification limits, yet exhibit rapid non-linear drift leading to catastrophic field failure ($168h$ burn-in completion).

This pipeline provides a two-stage screening architecture:
1. **Module A (Dynamic Outlier Detection)**: Identifies multi-dimensional contextual outliers at early burn-in ($0h \to 24h$) using unsupervised/semi-supervised anomaly detection, optimized via cost-sensitive $F_\beta$ thresholding ($\beta \ge 2$) to eliminate False Negatives (escapes).
2. **Module B (Time-Series Drift Predictor)**: Accurately forecasts late-stage degradation ($Value_{168h}$) using only early-life measurements ($Value_{0h}, Value_{24h}, \Delta_{0-24h}$), trained with Mean Absolute Error (L1 Loss) and coupled with Bayesian Ridge uncertainty bounds (95% UCL) for early abort screening.
3. **Explainability & QA Disposition Layer**: Generates SHAP local feature attributions and transparent surrogate decision tree `IF-THEN` rules for quality assurance inspectors.

---

## 🗂️ Project File Structure

```
ISRO/
├── data/
│   ├── raw/                                 # Storage for raw datasets
│   ├── processed/                           # Processed feature-engineered sets
│   └── generate_sample_data.py              # Realistic multi-lot burn-in data synthesizer
├── src/
│   ├── __init__.py                          # Package identifier
│   ├── feature_engineering.py               # Deltas (Δ_0-24h), Lot Z-Scores, Acceleration factors
│   ├── module_a_outlier_detection.py        # Mahalanobis, Isolation Forest, LOF, F_beta optimizer
│   ├── module_b_drift_predictor.py          # Gradient Boosting (MAE loss) + Bayesian Ridge (UCL)
│   ├── explainability.py                    # SHAP attribution + Surrogate IF-THEN rule extractor
│   └── evaluation.py                        # Cost-sensitive classification & MAE regression metrics
├── notebooks/
│   └── burn_in_anomaly_and_drift_pipeline.ipynb # Complete interactive Jupyter Notebook
├── tests/
│   └── test_pipeline.py                     # Automated integration & unit tests
├── requirements.txt                         # Pinned Python package dependencies
└── README.md                                # System documentation
```

---

## 🔬 Dataset Mappings

| Module | Benchmark Reference | Real-World Application in Pipeline |
| :--- | :--- | :--- |
| **Module A (Dynamic Outliers)** | **UCI SECOM Dataset** | Models high-dimensional sensor data with extreme class imbalance (~6% failure rate), noise, and intra-lot parametric distributions. |
| **Module A (Explainability)** | **Semiconductor Wafer Defect Dataset** | Uses explicit physical parameters (`leakage_current_ua`, `standby_current_ma`, `v_threshold_v`, `chamber_temp_c`, `stress_voltage_v`) for localized SHAP attribution. |
| **Module B (Drift Predictor)** | **NASA C-MAPSS Turbofan Degradation** | Maps run-to-failure degradation physics to burn-in intervals ($0h \to 24h \to 96h \to 168h$), predicting late-stage runaway drift from early velocity. |

---

## ⚙️ Core Technical Modules

### 1. Feature Engineering (`src/feature_engineering.py`)
- **Delta Features**: $\Delta_{0-24} = Value_{24h} - Value_{0h}$ and percentage drift rates $\frac{\Delta_{0-24}}{|Value_{0h}| + \epsilon}$.
- **Lot-Level Context**: Intra-batch statistical normalization ($Z\text{-score} = \frac{Value - \mu_{lot}}{\sigma_{lot}}$). Directly flags anomalies like a $45\mu A$ die in a $10\mu A$ baseline lot ($Z \approx +10.0$).
- **Acceleration Factors**: Stress interactions between oven temperature ($125^\circ\text{C}$), bias voltage ($3.6\text{V}$), and early drift rate ($\text{Stress} \times \Delta_{0-24}$).

### 2. Module A: Dynamic Outlier Detection (`src/module_a_outlier_detection.py`)
- **Mahalanobis Distance**: Covariance-aware statistical distance factoring in correlations between physical channels (e.g., leakage vs standby current). Uses Ledoit-Wolf shrinkage covariance.
- **Isolation Forest**: Efficient tree-based recursive partitioning isolating abnormal multidimensional points.
- **Local Outlier Factor (LOF)**: Density-based local neighborhood deviation scoring.
- **Ensemble Fusion & $F_\beta$ Calibration**: Normalizes and fuses scores; tunes decision boundary to maximize $F_2$ or $F_3$ score, suppressing False Negatives (zero escape policy).

### 3. Module B: Time-Series Drift Predictor (`src/module_b_drift_predictor.py`)
- **Gradient Boosting (XGBoost / HistGBR)**: Configured with **MAE (L1 loss)** to predict median degradation without overfitting to extreme non-linear outliers.
- **Bayesian Ridge Regression**: Provides full posterior predictive distributions $(\mu, \sigma)$. Calculates the 95% Upper Confidence Limit ($UCL = \mu + 1.96\sigma$). Dies exceeding safety limits at $168h$ ($UCL > 25\mu A$) are flagged for early test termination, saving energy and chamber capacity.

### 4. Explainability & QA Rules (`src/explainability.py`)
- **SHAP Waterfall & Attribution**: Quantifies exact mV, $\mu$A, or Z-score contributions for every flagged component.
- **Surrogate Decision Tree**: Converts ensemble black-box predictions into human-readable `IF-THEN` rules for quality engineers.

---

## 🚀 Quick Start Guide

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```powershell
python tests/test_pipeline.py
```

### 3. Generate Burn-In Dataset
```powershell
python data/generate_sample_data.py --n_samples 5000 --n_lots 50 --outlier_ratio 0.06
```

### 4. Direct Training on Real Kaggle & UCI Datasets
The pipeline includes automatic download and preprocessing for the exact benchmark datasets cited:
- **UCI SECOM (Kaggle `paresh2047/uci-semcom`)**: 1,567 real semiconductor production wafers, 591 sensors, exactly 104 defect failures.
- **NASA C-MAPSS (Kaggle `behdadk/nasa-cmaps`)**: Multi-variate time-series run-to-failure degradation trajectories.

#### Run Dedicated Kaggle Training:
```powershell
python train_on_kaggle_data.py
```

#### Dedicated Kaggle Jupyter Notebook:
Open [`notebooks/kaggle_burn_in_training_pipeline.ipynb`](file:///c:/Users/sunet/Documents/ISRO/notebooks/kaggle_burn_in_training_pipeline.ipynb) in VS Code, Antigravity IDE, or JupyterLab.
All 16 cells are pre-executed with:
- 4-panel real SECOM confusion matrices (Raw counts, Recall %, Precision %, Cost matrix)
- Real SECOM ROC & Precision-Recall curves
- C-MAPSS MAE iteration loss convergence curve
- C-MAPSS 4-panel residual error diagnostics
- Formal QA Screening & Lot Qualification Report (`reports/kaggle_training_report.md`)

---

### 5. Interactive Synthetic Burn-In Notebook
For multi-lot spatial contextual anomaly analysis ($10\mu A$ lot mean vs $45\mu A$ outlier) and SHAP explainability:
```powershell
jupyter notebook notebooks/burn_in_anomaly_and_drift_pipeline.ipynb
```

---

## 📑 Formal Reports & Defense Presentation

- **Comprehensive Project Final Report**: [`reports/PROJECT_FINAL_REPORT.md`](file:///c:/Users/sunet/Documents/ISRO/reports/PROJECT_FINAL_REPORT.md) (Full academic/industrial defense, mathematical derivations, empirical evaluation tables, and QA audit certificate).
- **Master Technical Blueprint**: [`reports/PS_COMPLETE_SOLUTION_BLUEPRINT.md`](file:///c:/Users/sunet/Documents/ISRO/reports/PS_COMPLETE_SOLUTION_BLUEPRINT.md) (Flowcharts, UVP, Challenges vs Mitigations, Others vs Ours matrix).
- **SIH 2026 Pitch Deck (6 Slides)**: [`NMIET_SIH_2026_PPT_Template_Updated.pptx`](file:///c:/Users/sunet/Documents/ISRO/NMIET_SIH_2026_PPT_Template_Updated.pptx) & [`NMIET_SIH_2026_PPT_Template.pptx`](file:///c:/Users/sunet/Documents/ISRO/NMIET_SIH_2026_PPT_Template.pptx).
- **Kaggle Benchmark Audit Report**: [`reports/kaggle_training_report.md`](file:///c:/Users/sunet/Documents/ISRO/reports/kaggle_training_report.md).
