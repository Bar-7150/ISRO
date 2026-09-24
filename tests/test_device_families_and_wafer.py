"""
tests/test_device_families_and_wafer.py
Validates the high-tech multi-device families, circular wafer generator, and PINN drift calculations.
"""

import unittest
from src.api_server import DEVICE_FAMILIES, generate_circular_wafer
from src.module_b_drift_predictor import PhysicsInformedDriftPredictor, calculate_arrhenius_acceleration


class TestHighTechEnhancements(unittest.TestCase):

    def test_device_families(self):
        self.assertEqual(len(DEVICE_FAMILIES), 5)
        for key in ["digital_cmos", "analog_opamp", "voltage_reference", "mems_gyro", "cmos_image_sensor"]:
            self.assertIn(key, DEVICE_FAMILIES)
            fam = DEVICE_FAMILIES[key]
            self.assertIn("name", fam)
            self.assertIn("standard", fam)
            self.assertIn("physics_equation", fam)
            self.assertIn("ea_ev", fam)
            self.assertGreater(fam["ea_ev"], 0.4)
            self.assertLess(fam["ea_ev"], 1.2)

    def test_circular_wafer_generator(self):
        dies = generate_circular_wafer(radius_dies=5, lot_mean=10.0, lot_std=2.5, dpat_limit=17.5)
        self.assertGreater(len(dies), 60)
        
        has_dpat_reject = any(d["is_dpat_reject"] for d in dies)
        has_gdbn_risk = any(d["is_gdbn_risk"] for d in dies)
        has_pass = any(not d["is_dpat_reject"] and not d["is_gdbn_risk"] for d in dies)
        
        self.assertTrue(has_dpat_reject, "Circular wafer must contain DPAT outlier die")
        self.assertTrue(has_gdbn_risk, "Circular wafer must contain GDBN spatial risk die")
        self.assertTrue(has_pass, "Circular wafer must contain qualified passing dies")

    def test_arrhenius_kinetics(self):
        af_cmos = calculate_arrhenius_acceleration(125.0, 25.0, ea_ev=0.70)
        af_opamp = calculate_arrhenius_acceleration(125.0, 25.0, ea_ev=0.62)
        self.assertGreater(af_cmos, 10.0)
        self.assertGreater(af_cmos, af_opamp)

    def test_pinn_drift_trajectory(self):
        predictor = PhysicsInformedDriftPredictor(safety_limit_168h=25.0, alpha_conformal=0.001)
        res = predictor.forecast_trajectory(val_0h=10.0, val_24h=18.5, temp_c=125.0, voltage_v=3.6)
        
        self.assertIn("pred_168h_point_ua", res)
        self.assertIn("pred_168h_conformal_upper_ua", res)
        self.assertIn("safety_slope_analysis", res)
        self.assertTrue(res["safety_slope_analysis"]["early_abort_triggered"])
        self.assertEqual(res["safety_slope_analysis"]["burn_in_hours_saved"], 144.0)


if __name__ == "__main__":
    unittest.main()
