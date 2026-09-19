"""
feature_engineering.py
Transforms raw burn-in parametric sensor data (0h, 24h, 96h, 168h) into
domain-informed features for anomaly detection and drift prediction:
1. Delta Features (Parametric shifts: Δ_0-24h, % drift)
2. Lot-Level Context (Mean, Std, Within-Lot Z-Scores)
3. Stress Acceleration Factors (Thermal & Voltage interaction)
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class BurnInFeatureEngineer:
    """
    Feature engineering pipeline for high-reliability semiconductor burn-in data.
    
    Transforms 0h and 24h test measurements into actionable signals capturing:
    - Intra-lot contextual anomalies (e.g. 45uA part in a 10uA lot)
    - Early degradation velocity (Delta 0-24h)
    - Acceleration stress interaction (Arrhenius/Eyring proxy models)
    """

    def __init__(
        self,
        lot_col: str = "lot_id",
        parametric_cols_0h: Optional[List[str]] = None,
        parametric_cols_24h: Optional[List[str]] = None,
        temp_col: str = "chamber_temp_c",
        voltage_col: str = "stress_voltage_v"
    ):
        self.lot_col = lot_col
        self.parametric_cols_0h = parametric_cols_0h or [
            "leakage_current_0h_ua",
            "standby_current_0h_ma",
            "v_threshold_0h_v"
        ]
        self.parametric_cols_24h = parametric_cols_24h or [
            "leakage_current_24h_ua",
            "standby_current_24h_ma",
            "v_threshold_24h_v"
        ]
        self.temp_col = temp_col
        self.voltage_col = voltage_col
        self.lot_stats_ = {}

    def fit(self, df: pd.DataFrame) -> "BurnInFeatureEngineer":
        """
        Computes and caches lot-level statistics (mean, std) from training data.
        """
        all_feature_cols = self.parametric_cols_0h + self.parametric_cols_24h
        
        # Calculate lot-level means and stds for all parametric columns
        stats = {}
        for col in all_feature_cols:
            if col in df.columns:
                group = df.groupby(self.lot_col)[col]
                stats[f"{col}_mean"] = group.transform("mean")
                stats[f"{col}_std"] = group.transform("std").replace(0, 1e-6)
                
        # Store global reference statistics in case unseen lots appear at inference
        self.global_means_ = {col: df[col].mean() for col in all_feature_cols if col in df.columns}
        self.global_stds_ = {col: max(df[col].std(), 1e-6) for col in all_feature_cols if col in df.columns}
        self.lot_stats_summary_ = (
            df.groupby(self.lot_col)[all_feature_cols]
            .agg(["mean", "std"])
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature transformations to the dataset.
        Returns a new DataFrame with engineered features appended.
        """
        X = df.copy()
        
        # 1. Delta Features (Early drift velocity between 0h and 24h)
        # Delta = Value_24h - Value_0h
        # Relative Delta = Delta / (Value_0h + eps)
        eps = 1e-6
        if "leakage_current_0h_ua" in X.columns and "leakage_current_24h_ua" in X.columns:
            X["delta_leakage_0_24_ua"] = X["leakage_current_24h_ua"] - X["leakage_current_0h_ua"]
            X["pct_drift_leakage_0_24"] = (
                X["delta_leakage_0_24_ua"] / (X["leakage_current_0h_ua"].abs() + eps)
            ) * 100.0

        if "standby_current_0h_ma" in X.columns and "standby_current_24h_ma" in X.columns:
            X["delta_standby_0_24_ma"] = X["standby_current_24h_ma"] - X["standby_current_0h_ma"]
            X["pct_drift_standby_0_24"] = (
                X["delta_standby_0_24_ma"] / (X["standby_current_0h_ma"].abs() + eps)
            ) * 100.0

        if "v_threshold_0h_v" in X.columns and "v_threshold_24h_v" in X.columns:
            X["delta_vth_0_24_v"] = X["v_threshold_24h_v"] - X["v_threshold_0h_v"]

        # 2. Lot-Level Context: Compute within-lot Z-scores
        # Value_0h_Zscore = (Value_0h - Lot_Mean_0h) / Lot_Std_0h
        # This addresses the contextual anomaly requirement: 45uA in a 10uA lot -> Z ~ +10.0!
        for col in self.parametric_cols_0h + self.parametric_cols_24h:
            if col in X.columns:
                # Lot-level mean & std calculation
                lot_mean = X.groupby(self.lot_col)[col].transform("mean")
                lot_std = X.groupby(self.lot_col)[col].transform("std").replace(0, 1e-6)
                
                # Fill missing lot stats with global fallback if lot has only 1 sample
                lot_mean = lot_mean.fillna(self.global_means_.get(col, 0.0))
                lot_std = lot_std.fillna(self.global_stds_.get(col, 1.0))
                
                X[f"{col}_lot_mean"] = lot_mean
                X[f"{col}_lot_std"] = lot_std
                X[f"{col}_zscore"] = (X[col] - lot_mean) / lot_std

        # Lot Z-score on early drift rate (Delta Z-score)
        if "delta_leakage_0_24_ua" in X.columns:
            delta_lot_mean = X.groupby(self.lot_col)["delta_leakage_0_24_ua"].transform("mean")
            delta_lot_std = X.groupby(self.lot_col)["delta_leakage_0_24_ua"].transform("std").replace(0, 1e-6)
            X["delta_leakage_0_24_zscore"] = (X["delta_leakage_0_24_ua"] - delta_lot_mean) / delta_lot_std

        # 3. Acceleration Factors (Thermal & Voltage Stress interactions)
        if self.temp_col in X.columns and self.voltage_col in X.columns:
            # Arrhenius thermal acceleration ratio proxy: Exp(-Ea / k * (1/T - 1/Tref))
            # Normalized around nominal 125C (398.15K) and 3.3V
            t_kelvin = X[self.temp_col] + 273.15
            t_ref_kelvin = 125.0 + 273.15
            # Simplified temperature stress multiplier: (T - T_nom) / 10
            temp_stress_factor = np.exp((t_kelvin - t_ref_kelvin) / 25.0)
            voltage_stress_factor = (X[self.voltage_col] / 3.3) ** 2.5
            
            X["thermal_accel_factor"] = temp_stress_factor
            X["voltage_accel_factor"] = voltage_stress_factor
            X["combined_stress_index"] = temp_stress_factor * voltage_stress_factor
            
            # Interaction term: Early drift velocity amplified by stress conditions
            if "delta_leakage_0_24_ua" in X.columns:
                X["stress_accelerated_drift"] = X["delta_leakage_0_24_ua"] * X["combined_stress_index"]

        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits statistics and transforms the data in one step."""
        return self.fit(df).transform(df)


def extract_feature_subsets(df: pd.DataFrame):
    """
    Returns feature column lists categorized for Module A and Module B.
    """
    # Module A features: Outlier detection using 0h parametric + lot context + early 24h delta
    module_a_features = [
        "leakage_current_0h_ua",
        "standby_current_0h_ma",
        "v_threshold_0h_v",
        "leakage_current_0h_ua_zscore",
        "standby_current_0h_ma_zscore",
        "v_threshold_0h_v_zscore",
        "delta_leakage_0_24_ua",
        "delta_leakage_0_24_zscore",
        "combined_stress_index"
    ]
    # Add sensor channels if present
    for sc in ["sensor_channel_1", "sensor_channel_2", "sensor_channel_3"]:
        if sc in df.columns:
            module_a_features.append(sc)

    # Module B features: Time-Series Drift Predictor inputs
    # Inputs: Value_0h, Value_24h, Delta_0-24h, Lot Context, Acceleration Factors
    # Target: Value_168h (leakage_current_168h_ua)
    module_b_features = [
        "leakage_current_0h_ua",
        "leakage_current_24h_ua",
        "delta_leakage_0_24_ua",
        "pct_drift_leakage_0_24",
        "leakage_current_0h_ua_zscore",
        "leakage_current_24h_ua_zscore",
        "delta_leakage_0_24_zscore",
        "standby_current_0h_ma",
        "standby_current_24h_ma",
        "v_threshold_0h_v",
        "v_threshold_24h_v",
        "chamber_temp_c",
        "stress_voltage_v",
        "combined_stress_index",
        "stress_accelerated_drift"
    ]
    
    # Filter features that exist in df
    available_a = [f for f in module_a_features if f in df.columns]
    available_b = [f for f in module_b_features if f in df.columns]
    return available_a, available_b
