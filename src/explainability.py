"""
explainability.py
Explainability & Interpretability Layer for High-Reliability QA Inspection.
1. SHAP (SHapley Additive exPlanations): Feature attribution and waterfall inspection
2. Surrogate Decision Tree Rule Extractor: Human-readable IF-THEN rules
3. Automated QA Disposition Report Generator
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Safe SHAP import with fallback
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class SurrogateRuleExtractor:
    """
    Extracts transparent, human-readable IF-THEN rules using an interpretable
    surrogate decision tree trained on the black-box ensemble / model decisions.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 15):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.surrogate_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        self.feature_names_ = []
        self.rules_text_ = ""

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> "SurrogateRuleExtractor":
        """
        Trains surrogate tree on model predictions to mimic black-box behavior.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values
        else:
            self.feature_names_ = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_mat = np.asarray(X)

        self.surrogate_tree.fit(X_mat, y_pred)
        self.rules_text_ = export_text(
            self.surrogate_tree,
            feature_names=self.feature_names_,
            decimals=3
        )
        return self

    def get_rules_text(self) -> str:
        """Returns the full IF-THEN tree rule hierarchy as formatted text."""
        return self.rules_text_

    def explain_sample_rule(
        self,
        sample: Union[pd.Series, np.ndarray, Dict]
    ) -> Dict[str, Union[str, float]]:
        """
        Traces the exact decision path through the surrogate tree for a single component,
        producing an inspector-grade rationale string.
        """
        if isinstance(sample, pd.Series):
            x_vec = sample[self.feature_names_].values.reshape(1, -1)
        elif isinstance(sample, dict):
            x_vec = np.array([sample[f] for f in self.feature_names_]).reshape(1, -1)
        else:
            x_vec = np.asarray(sample).reshape(1, -1)

        tree = self.surrogate_tree.tree_
        feature = tree.feature
        threshold = tree.threshold
        node_indicator = self.surrogate_tree.decision_path(x_vec)
        leaf_id = self.surrogate_tree.apply(x_vec)[0]
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

        conditions = []
        for node in node_index:
            if leaf_id == node:
                continue
            feat_idx = feature[node]
            feat_name = self.feature_names_[feat_idx]
            val = x_vec[0, feat_idx]
            thresh = threshold[node]
            if val <= thresh:
                conditions.append(f"{feat_name} <= {thresh:.3f} (measured: {val:.3f})")
            else:
                conditions.append(f"{feat_name} > {thresh:.3f} (measured: {val:.3f})")

        rule_str = " AND ".join(conditions) if conditions else "Baseline within normal lot limits"
        pred_class = int(self.surrogate_tree.predict(x_vec)[0])
        class_probs = self.surrogate_tree.predict_proba(x_vec)[0]
        confidence = float(class_probs[pred_class])

        disposition = "REJECT / QUARANTINE" if pred_class == 1 else "PASS / ACCEPT"
        return {
            "disposition": disposition,
            "confidence": confidence,
            "rule": rule_str
        }


class QAInspectionExplainer:
    """
    Combines SHAP feature importance attributions with human-readable surrogate rules.
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.surrogate = SurrogateRuleExtractor(max_depth=3)
        
        if HAS_SHAP:
            # Check if model has tree attribute or is scikit-learn/xgboost
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                if background_data is not None:
                    # Sample background for Kernel/Exact explainer
                    bg_sample = shap.sample(background_data, 50, random_state=42)
                    self.explainer = shap.Explainer(model.predict, bg_sample)

    def fit_surrogate_rules(self, X: pd.DataFrame, y_pred: np.ndarray):
        """Fits the surrogate decision tree and calibrates TreeExplainer."""
        self.surrogate.fit(X, y_pred, feature_names=self.feature_names)
        if HAS_SHAP and self.explainer is None:
            try:
                self.explainer = shap.TreeExplainer(self.surrogate.surrogate_tree)
            except Exception:
                pass

    def explain_component(
        self,
        component_data: Union[pd.Series, pd.DataFrame],
        die_id: str = "UNKNOWN_DIE",
        lot_id: str = "UNKNOWN_LOT"
    ) -> Dict:
        """
        Generates a comprehensive QA Inspection Briefing for a specific component die.
        """
        if isinstance(component_data, pd.DataFrame):
            row = component_data.iloc[0]
        else:
            row = component_data

        # 1. Surrogate Tree Rule
        rule_info = self.surrogate.explain_sample_rule(row)

        # 2. SHAP Values
        top_shap_factors = []
        if self.explainer is not None:
            x_mat = np.array([[row[f] for f in self.feature_names]])
            shap_values = self.explainer(x_mat)
            if hasattr(shap_values, "values"):
                vals = shap_values.values[0]
            else:
                vals = np.asarray(shap_values)[0]

            # Pair with feature names and sort by absolute contribution
            ranked_idx = np.argsort(np.abs(vals))[::-1]
            for idx in ranked_idx[:5]:
                f_name = self.feature_names[idx]
                top_shap_factors.append({
                    "feature": f_name,
                    "measured_value": float(row[f_name]),
                    "shap_impact": float(vals[idx]),
                    "direction": "Pushes toward REJECT" if vals[idx] > 0 else "Pushes toward PASS"
                })

        # Assemble QA report string
        report_lines = [
            f"=== SEMICONDUCTOR BURN-IN QA DISPOSITION BRIEFING ===",
            f"Component Die ID: {die_id} | Lot: {lot_id}",
            f"Final Screening Disposition: {rule_info['disposition']} (Surrogate Confidence: {rule_info['confidence']:.1%})",
            f"Primary Decision Boundary Rule:",
            f"   IF {rule_info['rule']} -> {rule_info['disposition']}",
        ]
        
        if top_shap_factors:
            report_lines.append("Top Root-Cause Sensor Driving Factors (SHAP attribution):")
            for i, f in enumerate(top_shap_factors, 1):
                report_lines.append(
                    f"   {i}. {f['feature']} = {f['measured_value']:.3f} | Impact: {f['shap_impact']:+.4f} ({f['direction']})"
                )
        report_lines.append("=" * 55)

        return {
            "die_id": die_id,
            "lot_id": lot_id,
            "disposition": rule_info["disposition"],
            "surrogate_rule": rule_info["rule"],
            "surrogate_confidence": rule_info["confidence"],
            "top_shap_factors": top_shap_factors,
            "qa_briefing_text": "\n".join(report_lines)
        }
