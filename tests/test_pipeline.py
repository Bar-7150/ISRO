"""
test_pipeline.py
Unit and integration tests for the Semiconductor Burn-In Anomaly & Drift Pipeline.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root and src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.generate_sample_data import generate_burn_in_dataset
from src.feature_engineering import BurnInFeatureEngineer, extract_feature_subsets
from src.module_a_outlier_detection import MahalanobisDetector, DynamicOutlierDetector
from src.module_b_drift_predictor import BurnInDriftForecaster
from src.explainability import SurrogateRuleExtractor, QAInspectionExplainer
from src.evaluation import (
    compute_cost_sensitive_classification_metrics,
    compute_drift_regression_metrics
)


class TestBurnInPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate a small synthetic batch for quick testing
        cls.raw_df = generate_burn_in_dataset(
            n_samples=200,
            n_lots=5,
            outlier_ratio=0.10,
            random_state=42
        )

    def test_01_data_generation(self):
        self.assertEqual(len(self.raw_df), 200)
        self.assertIn("lot_id", self.raw_df.columns)
        self.assertIn("leakage_current_0h_ua", self.raw_df.columns)
        self.assertIn("leakage_current_168h_ua", self.raw_df.columns)
        self.assertIn("is_defect_168h", self.raw_df.columns)
        self.assertTrue(self.raw_df["is_defect_168h"].sum() > 0)

    def test_02_feature_engineering(self):
        fe = BurnInFeatureEngineer()
        df_feat = fe.fit_transform(self.raw_df)
        
        # Verify Delta features
        self.assertIn("delta_leakage_0_24_ua", df_feat.columns)
        self.assertIn("pct_drift_leakage_0_24", df_feat.columns)
        
        # Verify Lot Z-Scores
        self.assertIn("leakage_current_0h_ua_zscore", df_feat.columns)
        self.assertIn("delta_leakage_0_24_zscore", df_feat.columns)
        
        # Verify Acceleration Factors
        self.assertIn("combined_stress_index", df_feat.columns)
        self.assertIn("stress_accelerated_drift", df_feat.columns)
        
        feat_a, feat_b = extract_feature_subsets(df_feat)
        self.assertTrue(len(feat_a) > 0)
        self.assertTrue(len(feat_b) > 0)

    def test_03_module_a_mahalanobis(self):
        fe = BurnInFeatureEngineer()
        df_feat = fe.fit_transform(self.raw_df)
        feat_a, _ = extract_feature_subsets(df_feat)
        X = df_feat[feat_a].values
        
        detector = MahalanobisDetector(regularize=True)
        detector.fit(X)
        scores = detector.score_samples(X)
        self.assertEqual(len(scores), len(X))
        self.assertTrue(np.all(scores >= 0))

    def test_04_module_a_dynamic_outlier_detector(self):
        fe = BurnInFeatureEngineer()
        df_feat = fe.fit_transform(self.raw_df)
        feat_a, _ = extract_feature_subsets(df_feat)
        X = df_feat[feat_a]
        y = df_feat["is_defect_168h"].values
        
        detector = DynamicOutlierDetector(n_estimators=30, n_neighbors=10)
        detector.fit(X)
        scores_df = detector.predict_anomaly_scores(X)
        
        self.assertIn("ensemble_anomaly_score", scores_df.columns)
        self.assertIn("mahalanobis_score", scores_df.columns)
        self.assertIn("iforest_score", scores_df.columns)
        self.assertIn("lof_score", scores_df.columns)
        
        # Test F_beta optimization
        opt_res = detector.optimize_fbeta_threshold(y, scores_df["ensemble_anomaly_score"].values, beta=2.0)
        self.assertIn("optimal_threshold", opt_res)
        self.assertGreater(opt_res["best_fbeta"], 0.0)

    def test_05_module_b_drift_predictor(self):
        fe = BurnInFeatureEngineer()
        df_feat = fe.fit_transform(self.raw_df)
        _, feat_b = extract_feature_subsets(df_feat)
        X = df_feat[feat_b]
        y = df_feat["leakage_current_168h_ua"]
        
        forecaster = BurnInDriftForecaster(safety_limit_168h=25.0)
        forecaster.fit(X, y)
        preds_df = forecaster.predict(X)
        
        self.assertIn("pred_168h_gbr", preds_df.columns)
        self.assertIn("pred_168h_bayes_mean", preds_df.columns)
        self.assertIn("pred_168h_ucl", preds_df.columns)
        self.assertIn("ucl_exceeds_safety_limit", preds_df.columns)
        
        eval_metrics = forecaster.evaluate(X, y)
        self.assertIn("gbr_mae", eval_metrics)
        self.assertIn("bayesian_ucl_coverage_prob", eval_metrics)

    def test_06_explainability_surrogate(self):
        fe = BurnInFeatureEngineer()
        df_feat = fe.fit_transform(self.raw_df)
        feat_a, _ = extract_feature_subsets(df_feat)
        X = df_feat[feat_a]
        y = df_feat["is_defect_168h"]
        
        extractor = SurrogateRuleExtractor(max_depth=3)
        extractor.fit(X, y)
        rules = extractor.get_rules_text()
        self.assertTrue(len(rules) > 10)
        
        single_exp = extractor.explain_sample_rule(X.iloc[0])
        self.assertIn("disposition", single_exp)
        self.assertIn("rule", single_exp)

    def test_07_evaluation_metrics(self):
        y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0])
        metrics = compute_cost_sensitive_classification_metrics(y_true, y_pred, beta=2.0)
        self.assertIn("recall (catch_rate)", metrics)
        self.assertIn("escape_rate (fn_rate)", metrics)
        self.assertIn("f2_score", metrics)


if __name__ == "__main__":
    unittest.main()
