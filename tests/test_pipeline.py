"""
test_pipeline.py
Unit and integration tests for the Semiconductor Burn-In Anomaly & Drift Pipeline.
"""

import os
import sys
import json
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

    def test_08_dpat_and_gdbn(self):
        from src.module_a_outlier_detection import DPATEngine, GDBNSpatialEngine

        # 1. Verify exact ISRO benchmark calculation (10uA lot, 45uA part, 50uA spec)
        bench = DPATEngine.evaluate_latent_defect_benchmark(
            lot_mean=10.0, lot_std=2.5, candidate_value=45.0, static_spec_limit=50.0, k_sigma=3.0
        )
        self.assertEqual(bench["dpat_upper_limit_ua"], 17.5)
        self.assertEqual(bench["z_score_sigma"], 14.0)
        self.assertTrue(bench["escapes_static_screening"])
        self.assertEqual(bench["static_screening_disposition"], "PASS (ESCAPE!)")
        self.assertEqual(bench["dpat_dynamic_disposition"], "REJECT (CAUGHT)")

        # 2. Verify GDBN spatial clustering calculation
        sample_tray = pd.DataFrame({
            "socket_x": [0, 0, 1, 1],
            "socket_y": [0, 1, 0, 1],
            "dpat_reject": [1, 1, 1, 0] # Die at index 3 is surrounded by 3 failing neighbors
        })
        gdbn = GDBNSpatialEngine(neighborhood_radius=1.5, defect_density_threshold=0.35)
        res = gdbn.compute_spatial_risk(sample_tray)
        self.assertIn("gdbn_spatial_reject", res.columns)
        self.assertEqual(res.iloc[3]["gdbn_neighbor_defects"], 3)
        self.assertEqual(res.iloc[3]["gdbn_spatial_reject"], 1)

    def test_09_physics_drift_and_safety_slope(self):
        from src.module_b_drift_predictor import (
            PhysicsInformedDriftPredictor,
            calculate_arrhenius_acceleration,
            calculate_electromigration_af,
            calculate_safety_slope
        )

        # 1. Verify Arrhenius thermal acceleration
        af_t = calculate_arrhenius_acceleration(temp_stress_c=125.0, temp_use_c=25.0, ea_ev=0.7)
        self.assertTrue(af_t > 50.0)

        # 2. Verify Electromigration acceleration
        af_em = calculate_electromigration_af(temp_stress_c=125.0, temp_use_c=25.0, v_stress=3.6, v_nominal=3.3)
        self.assertTrue(af_em > af_t)

        # 3. Verify safety slope and early 24h abort trigger
        slope = calculate_safety_slope(val_0h=10.0, val_24h=20.0, pred_168h=40.0, pred_168h_ucl=45.0, safety_limit_168h=25.0)
        self.assertTrue(slope["early_abort_triggered"])
        self.assertEqual(slope["burn_in_hours_saved"], 144.0)
        self.assertAlmostEqual(slope["energy_savings_pct"], 85.71, places=1)

    def test_10_hardware_controller_and_relay(self):
        from src.hardware_edge_controller import HardwareEdgeController

        controller = HardwareEdgeController()
        self.assertTrue(controller.dut_powered)
        self.assertEqual(controller.relay_state, "CLOSED_POWER_ON")

        # Read sensors
        telemetry = controller.read_sensors()
        self.assertIn("chamber_temp_c", telemetry)
        self.assertIn("rail_voltage_v", telemetry)
        self.assertIn("iddq_leakage_current_ua", telemetry)

        # Actuate relay trip
        controller.trip_relay(reason="Unit Test Trip")
        self.assertFalse(controller.dut_powered)
        self.assertEqual(controller.relay_state, "OPEN_POWER_CUT")
        post_trip = controller.read_sensors()
        self.assertEqual(post_trip["iddq_leakage_current_ua"], 0.0)

        # Reset relay
        controller.reset_relay()
        self.assertTrue(controller.dut_powered)
        self.assertEqual(controller.relay_state, "CLOSED_POWER_ON")

    def test_11_conformal_prediction_and_risk_bound(self):
        from src.module_b_drift_predictor import ConformalDriftPredictor, PhysicsInformedDriftPredictor

        # Synthetic calibration & evaluation data
        rng = np.random.RandomState(42)
        X_train = rng.randn(60, 4)
        y_train = 10.0 + 2.0 * X_train[:, 0] + rng.randn(60) * 0.3
        X_cal = rng.randn(40, 4)
        y_cal = 10.0 + 2.0 * X_cal[:, 0] + rng.randn(40) * 0.3
        X_test = rng.randn(20, 4)
        y_test = 10.0 + 2.0 * X_test[:, 0] + rng.randn(20) * 0.3

        predictor = ConformalDriftPredictor(alpha=0.001, random_state=42)
        predictor.fit(X_train, y_train)
        predictor.calibrate(X_cal, y_cal)
        self.assertTrue(predictor.is_calibrated_)

        preds = predictor.predict_conformal(X_test)
        self.assertEqual(len(preds), 20)
        self.assertIn("conformal_upper_bound", preds.columns)
        self.assertIn("conformal_lower_bound", preds.columns)
        self.assertEqual(preds["confidence_level_pct"].iloc[0], 99.9)

        # Verify mathematical coverage guarantee: almost all test samples should fall below upper conformal limit
        empirical_coverage = np.mean(y_test <= preds["conformal_upper_bound"].values)
        self.assertGreaterEqual(empirical_coverage, 0.95)

    def test_12_arrhenius_pinn_loss_and_floors(self):
        from src.module_b_drift_predictor import (
            calculate_arrhenius_thermo_floor,
            compute_pinn_arrhenius_loss
        )

        v0 = np.array([10.0, 12.0])
        v24 = np.array([14.0, 16.0])
        y_true = np.array([28.0, 32.0])

        # 1. Verify thermodynamic floor calculation
        floor = calculate_arrhenius_thermo_floor(v0, v24, temp_c=125.0)
        self.assertTrue(np.all(floor > v24))

        # 2. Test unphysically flat prediction (violating thermodynamics)
        y_flat = np.array([11.0, 13.0])  # Flatter than physics dictates
        loss_flat = compute_pinn_arrhenius_loss(y_true, y_flat, v0, v24, temp_c=125.0, lambda_arrhenius=2.0)
        self.assertFalse(loss_flat["thermo_compliant"])
        self.assertGreater(loss_flat["loss_arrhenius_penalty"], 0.0)

        # 3. Test physically compliant prediction (above thermodynamic floor)
        y_compliant = np.array([70.0, 75.0])
        y_true_comp = np.array([72.0, 74.0])
        loss_comp = compute_pinn_arrhenius_loss(y_true_comp, y_compliant, v0, v24, temp_c=125.0, lambda_arrhenius=2.0)
        self.assertTrue(loss_comp["thermo_compliant"])
        self.assertEqual(loss_comp["arrhenius_violation_rate"], 0.0)

    def test_13_sensor_auto_zero_and_thermal_compensation(self):
        from src.hardware_edge_controller import HardwareEdgeController

        controller = HardwareEdgeController()
        
        # 1. Verify auto-zero baseline calibration
        cal_profile = controller.auto_zero_calibrate(ambient_temp_c=25.0)
        self.assertEqual(cal_profile["status"], "CALIBRATED")
        self.assertEqual(cal_profile["calibration_temp_c"], 25.0)
        self.assertTrue(controller.is_calibrated)

        # 2. Verify thermal compensation math when ambient box temperature heats to 60C
        raw_ua = 150.0  # Raw reading with substantial sensor drift
        box_temp_c = 60.0
        comp_ua, drift_ua = controller.apply_thermal_compensation(raw_ua, box_temp_c)
        self.assertLess(comp_ua, raw_ua)
        self.assertGreater(drift_ua, 0.0)

        # 3. Read sensors telemetry and verify fields
        telemetry = controller.read_sensors()
        self.assertIn("raw_iddq_leakage_current_ua", telemetry)
        self.assertIn("iddq_leakage_current_ua", telemetry)
        self.assertIn("sensor_thermal_drift_ua", telemetry)
        self.assertTrue(telemetry["sensor_thermal_compensated"])

    def test_14_as9100_digital_birth_certificate(self):
        from src.as9100_compliance import AS9100CertificateGenerator

        gen = AS9100CertificateGenerator(output_dir="reports/birth_certificates")
        cert = gen.create_certificate(
            chip_serial_id="TEST-CHIP-AS9100-001",
            socket_id="CHAMBER-01/TRAY-01/SOCKET-A1",
            lot_mean=10.0,
            lot_std=2.0,
            val_0h=9.8,
            val_24h=10.2,
            pred_168h_point=11.1,
            conformal_upper_168h=12.5
        )

        # 1. Verify cryptographic SHA-256 signature
        self.assertIn("cryptographic_verification", cert)
        self.assertTrue(gen.verify_integrity(cert))

        # 2. Verify tamper detection
        tampered_cert = json.loads(json.dumps(cert))
        tampered_cert["telemetry_and_predictions"]["val_0h_measured_ua"] = 999.0
        self.assertFalse(gen.verify_integrity(tampered_cert))

        # 3. Export JSON and PDF
        json_path = gen.export_json(cert, filename="test_cert.json")
        self.assertTrue(os.path.exists(json_path))

        pdf_path = gen.export_pdf(cert, filename="test_cert.pdf")
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)

    def test_15_federated_edge_mesh_and_fedavg(self):
        from src.federated_edge_mesh import FederatedEdgeNode, FederatedMeshCoordinator

        coordinator = FederatedMeshCoordinator()
        node1 = FederatedEdgeNode("CHAMBER-01", "Bengaluru Cleanroom", dp_epsilon=0.5)
        node2 = FederatedEdgeNode("CHAMBER-02", "Sriharikota QA Bay", dp_epsilon=0.5)

        coordinator.register_node(node1)
        coordinator.register_node(node2)
        self.assertEqual(len(coordinator.nodes), 2)

        # Simulate sessions
        node1.record_burn_in_session(n_chips=100, hours_tested=24.0, measured_mean_slope=0.013, measured_ea=0.705)
        node2.record_burn_in_session(n_chips=100, hours_tested=24.0, measured_mean_slope=0.014, measured_ea=0.710)

        # Check anonymized update payload privacy
        update = node1.generate_anonymized_update(coordinator.global_parameters)
        self.assertIn("Differential Privacy", update["privacy_guarantee"])
        self.assertIn("anonymized_parameter_deltas", update)

        # Execute FedAvg round
        initial_round = coordinator.current_round
        round_summary = coordinator.execute_federated_round()
        self.assertEqual(round_summary["round_number"], initial_round + 1)
        self.assertGreater(round_summary["total_mesh_test_hours"], 0.0)
        self.assertEqual(round_summary["participating_nodes_count"], 2)


if __name__ == "__main__":
    unittest.main()


