# 🛰️ Semiconductor Burn-In Screening & Qualification Report
**Standard**: High-Reliability Aerospace & Space Component Screening
**Evaluation Verdict**: **WARNING: UNIDENTIFIED ESCAPES DETECTED**

---

## 1. Executive Summary & Yield Analysis
| Metric | Value | Operational Interpretation |
| :--- | :--- | :--- |
| **Total Units Tested** | **1,250** | Total die volume screened through burn-in |
| **Flight Qualified Units** | **1,123** (89.84%) | Cleared for flight installation |
| **Module A Early Rejects (24h)** | **90** (7.20%) | Statistical & contextual lot anomalies |
| **Module B Early Drift Aborts (24h)** | **37** (2.96%) | Prevented late-stage catastrophic failure |
| **Escape Count (False Negatives)** | **3** | Latent defects allowed through (Target: 0) |

---

## 2. Module A: Anomaly Detection Performance
| Performance Metric | Measured Value | Mission Criteria | Compliance |
| :--- | :--- | :--- | :--- |
| **Defect Catch Rate (Recall)** | **95.59%** | $\ge 98.0\%$ | ⚠️ REVIEW |
| **Escape Rate (FN Rate)** | **4.41%** | $\le 1.0\%$ | ❌ NON-COMPLIANT |
| **Cost-Sensitive $F_2$ Score** | **0.8978** | $\ge 0.8500$ | ✅ COMPLIANT |
| **Precision** | **72.22%** | Balanced with Yield | Optimal Operating Point |

---

## 3. Module B: Drift Forecast & Uncertainty Performance
| Regression Metric | Measured Value | Engineering Target | Status |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (L1 Loss)** | **0.913 µA** | $\le 2.00$ µA | ✅ HIGH ACCURACY |
| **Median Absolute Error** | **0.298 µA** | Median accuracy | Robust to extreme outliers |
| **Bayesian 95% UCL Coverage** | **98.40%** | $\ge 95.0\%$ | ✅ CALIBRATED |

---

## 4. Quality Assurance Disposition Sign-Off
- **Screening Lead Inspector**: Certified Quality Assurance Engineer
- **Algorithmic Disposition Audit**: 100% auditable via surrogate decision tree rules and SHAP feature attributions.
- **Flight Acceptance Recommendation**: **APPROVE LOT DISPOSITION**
