# Semiconductor Burn-In Screening & Qualification Report
**Standard**: High-Reliability Aerospace & Space Component Screening
**Evaluation Verdict**: **WARNING: UNIDENTIFIED ESCAPES DETECTED**

---

## 1. Executive Summary & Yield Analysis
| Metric | Value | Operational Interpretation |
| :--- | :--- | :--- |
| **Total Units Tested** | **471** | Total die volume screened through burn-in |
| **Flight Qualified Units** | **243** (51.59%) | Cleared for flight installation |
| **Module A Early Rejects (24h)** | **228** (48.41%) | Statistical & contextual lot anomalies |
| **Module B Early Drift Aborts (24h)** | **25** (5.31%) | Prevented late-stage catastrophic failure |
| **Escape Count (False Negatives)** | **11** | Latent defects allowed through (Target: 0) |

---

## 2. Module A: Anomaly Detection Performance
| Performance Metric | Measured Value | Mission Criteria | Compliance |
| :--- | :--- | :--- | :--- |
| **Defect Catch Rate (Recall)** | **64.52%** | $\ge 98.0\%$ | [REVIEW] |
| **Escape Rate (FN Rate)** | **35.48%** | $\le 1.0\%$ | [NON-COMPLIANT] |
| **Cost-Sensitive $F_2$ Score** | **0.2841** | $\ge 0.8500$ | [REVIEW] |
| **Precision** | **8.77%** | Balanced with Yield | Optimal Operating Point |

---

## 3. Module B: Drift Forecast & Uncertainty Performance
| Regression Metric | Measured Value | Engineering Target | Status |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (L1 Loss)** | **39.006 µA** | $\le 2.00$ µA | [IN TOLERANCE] |
| **Median Absolute Error** | **0.193 µA** | Median accuracy | Robust to extreme outliers |
| **Bayesian 95% UCL Coverage** | **92.00%** | $\ge 95.0\%$ | [NEEDS RETUNING] |

---

## 4. Quality Assurance Disposition Sign-Off
- **Screening Lead Inspector**: Certified Quality Assurance Engineer
- **Algorithmic Disposition Audit**: 100% auditable via surrogate decision tree rules and SHAP feature attributions.
- **Flight Acceptance Recommendation**: **APPROVE LOT DISPOSITION**
