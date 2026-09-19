"""
generate_sample_data.py
Synthesizes high-reliability semiconductor burn-in test data mimicking:
1. UCI SECOM (extreme class imbalance, sensor noise, parametric distributions)
2. Semiconductor Wafer Physical Parameters (leakage_current_ua, standby_current_ma,
   v_threshold_v, etch_rate_nm_min, chamber_temp_c, stress_voltage_v)
3. NASA C-MAPSS Degradation (time-series drift progression across 0h, 24h, 96h, and 168h)
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_burn_in_dataset(
    n_samples: int = 5000,
    n_lots: int = 50,
    outlier_ratio: float = 0.06,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a synthetic semiconductor burn-in dataset.
    
    Parameters:
    -----------
    n_samples : int
        Total number of components/dies tested.
    n_lots : int
        Number of production lots/batches.
    outlier_ratio : float
        Proportion of anomalous/latent defect parts (~6% mirroring SECOM).
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame: Tabular burn-in dataset with 0h, 24h, 96h, and 168h measurements.
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Generate Lot IDs and baseline lot variations
    # Different lots have different baseline manufacturing process centers
    samples_per_lot = n_samples // n_lots
    lot_ids = [f"LOT_{i:03d}" for i in range(1, n_lots + 1)]
    
    # Lot-level baseline parameters (wafer fab drift between batches)
    lot_leakage_mean = {lot: rng.uniform(8.0, 15.0) for lot in lot_ids}
    lot_vth_mean = {lot: rng.uniform(0.68, 0.74) for lot in lot_ids}
    lot_etch_mean = {lot: rng.uniform(480.0, 520.0) for lot in lot_ids}
    
    data_records = []
    
    for i in range(n_samples):
        lot = lot_ids[i % n_lots]
        die_id = f"{lot}_DIE_{i:05d}"
        
        # Determine if this unit has a latent defect (imbalanced)
        is_latent_defect = rng.rand() < outlier_ratio
        
        # Manufacturing Physical Parameters
        base_leakage_mean = lot_leakage_mean[lot]
        base_vth_mean = lot_vth_mean[lot]
        base_etch = lot_etch_mean[lot]
        
        # Chamber Stress Conditions (Burn-In acceleration factors)
        chamber_temp_c = rng.normal(125.0, 1.2) # Nominal 125C burn-in oven
        stress_voltage_v = rng.normal(3.6, 0.04) # Nominal 3.6V accelerated stress
        etch_rate_nm_min = rng.normal(base_etch, 10.0)
        
        if not is_latent_defect:
            # Normal Unit: Tight distribution around lot mean
            # 0h Measurements
            leakage_0h = rng.normal(base_leakage_mean, 1.2)
            standby_0h = rng.normal(2.5, 0.15) # mA
            vth_0h = rng.normal(base_vth_mean, 0.015) # V
            
            # Drift progression across burn-in: Minor stable aging drift
            # 24h: subtle settling
            drift_24 = rng.normal(0.4, 0.15)
            leakage_24h = leakage_0h + drift_24
            standby_24h = standby_0h + rng.normal(0.02, 0.01)
            vth_24h = vth_0h - rng.normal(0.003, 0.001)
            
            # 96h & 168h: Logarithmic / stabilized degradation
            drift_96 = drift_24 + rng.normal(0.8, 0.25)
            leakage_96h = leakage_0h + drift_96
            
            drift_168 = drift_96 + rng.normal(0.6, 0.2)
            leakage_168h = leakage_0h + drift_168
            standby_168h = standby_0h + rng.normal(0.05, 0.02)
            vth_168h = vth_0h - rng.normal(0.008, 0.002)
            
            failure_flag_168h = 0
            defect_type = "PASS"
            
        else:
            # Latent Defect Unit:
            # Case 1: Contextual Outlier at 0h (e.g. 45 uA in a 10 uA lot)
            # Case 2: In-spec at 0h, but abnormal acceleration/drift at 24h
            anomaly_subtype = rng.choice(["contextual_0h", "early_drift_24h", "accelerated_thermal"])
            
            if anomaly_subtype == "contextual_0h":
                # Significant deviation relative to lot mean, even if within generic global spec (< 50uA)
                leakage_0h = base_leakage_mean + rng.uniform(25.0, 35.0) # e.g. ~ 40-45 uA
                standby_0h = rng.normal(3.8, 0.3)
                vth_0h = rng.normal(base_vth_mean - 0.04, 0.02)
                
                # Rapid degradation during burn-in
                leakage_24h = leakage_0h + rng.uniform(8.0, 18.0)
                standby_24h = standby_0h + rng.uniform(0.5, 1.2)
                vth_24h = vth_0h - rng.uniform(0.02, 0.05)
                
                leakage_96h = leakage_24h + rng.uniform(20.0, 45.0)
                leakage_168h = leakage_96h + rng.uniform(30.0, 70.0) # Catastrophic drift
                standby_168h = standby_24h + rng.uniform(1.0, 2.5)
                vth_168h = vth_24h - rng.uniform(0.04, 0.1)
                defect_type = "LATENT_CONTEXTUAL_OUTLIER"

            elif anomaly_subtype == "early_drift_24h":
                # Starts close to normal at 0h, but oxide breakdown causes rapid early drift at 24h
                leakage_0h = rng.normal(base_leakage_mean + 1.5, 1.5)
                standby_0h = rng.normal(2.6, 0.2)
                vth_0h = rng.normal(base_vth_mean, 0.015)
                
                # Sharp jump at 24h
                leakage_24h = leakage_0h + rng.uniform(7.0, 15.0)
                standby_24h = standby_0h + rng.uniform(0.3, 0.7)
                vth_24h = vth_0h - rng.uniform(0.015, 0.03)
                
                leakage_96h = leakage_24h + rng.uniform(18.0, 35.0)
                leakage_168h = leakage_96h + rng.uniform(25.0, 50.0)
                standby_168h = standby_24h + rng.uniform(0.8, 1.8)
                vth_168h = vth_24h - rng.uniform(0.03, 0.07)
                defect_type = "LATENT_DRIFT_RUNAWAY"

            else: # accelerated_thermal
                # High sensitivity to burn-in thermal/voltage stress
                leakage_0h = rng.normal(base_leakage_mean + 2.0, 1.5)
                standby_0h = rng.normal(2.8, 0.25)
                vth_0h = rng.normal(base_vth_mean, 0.015)
                
                thermal_factor = (chamber_temp_c - 120.0) * 0.8 + (stress_voltage_v - 3.5) * 12.0
                leakage_24h = leakage_0h + rng.uniform(5.0, 10.0) + thermal_factor
                standby_24h = standby_0h + rng.uniform(0.2, 0.5)
                vth_24h = vth_0h - rng.uniform(0.01, 0.02)
                
                leakage_96h = leakage_24h + rng.uniform(15.0, 30.0)
                leakage_168h = leakage_96h + rng.uniform(20.0, 45.0)
                standby_168h = standby_24h + rng.uniform(0.5, 1.5)
                vth_168h = vth_24h - rng.uniform(0.02, 0.05)
                defect_type = "THERMAL_VOLTAGE_STRESS_FAILURE"

            failure_flag_168h = 1
            
        # Add slight SECOM-like sensor measurement noise
        leakage_0h = max(0.1, round(leakage_0h + rng.normal(0, 0.05), 3))
        leakage_24h = max(0.1, round(leakage_24h + rng.normal(0, 0.08), 3))
        leakage_96h = max(0.1, round(leakage_96h + rng.normal(0, 0.12), 3))
        leakage_168h = max(0.1, round(leakage_168h + rng.normal(0, 0.15), 3))
        
        # Parametric sensor channels (representing SECOM 591 channels subset)
        sensor_channel_1 = round(rng.normal(100.0, 5.0) + (15.0 if is_latent_defect else 0.0), 2)
        sensor_channel_2 = round(rng.normal(45.0, 2.5) - (8.0 if is_latent_defect else 0.0), 2)
        sensor_channel_3 = round(rng.normal(1.25, 0.08), 3)

        data_records.append({
            "die_id": die_id,
            "lot_id": lot,
            "chamber_temp_c": round(chamber_temp_c, 2),
            "stress_voltage_v": round(stress_voltage_v, 3),
            "etch_rate_nm_min": round(etch_rate_nm_min, 2),
            "leakage_current_0h_ua": leakage_0h,
            "standby_current_0h_ma": round(standby_0h, 3),
            "v_threshold_0h_v": round(vth_0h, 4),
            "leakage_current_24h_ua": leakage_24h,
            "standby_current_24h_ma": round(standby_24h, 3),
            "v_threshold_24h_v": round(vth_24h, 4),
            "leakage_current_96h_ua": leakage_96h,
            "leakage_current_168h_ua": leakage_168h,
            "standby_current_168h_ma": round(standby_168h, 3),
            "v_threshold_168h_v": round(vth_168h, 4),
            "sensor_channel_1": sensor_channel_1,
            "sensor_channel_2": sensor_channel_2,
            "sensor_channel_3": sensor_channel_3,
            "defect_type": defect_type,
            "is_defect_168h": failure_flag_168h
        })

    df = pd.DataFrame(data_records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic semiconductor burn-in dataset.")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of component dies")
    parser.add_argument("--n_lots", type=int, default=50, help="Number of fabrication lots")
    parser.add_argument("--outlier_ratio", type=float, default=0.06, help="Defect ratio (approx 6%)")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "burn_in_semiconductor_data.csv")
    
    print(f"Generating {args.n_samples} burn-in records across {args.n_lots} lots...")
    df = generate_burn_in_dataset(
        n_samples=args.n_samples,
        n_lots=args.n_lots,
        outlier_ratio=args.outlier_ratio
    )
    df.to_csv(out_path, index=False)
    print(f"Saved dataset successfully to: {out_path}")
    print(f"Dataset summary: Shape={df.shape}, Defects={df['is_defect_168h'].sum()} ({df['is_defect_168h'].mean():.2%})")


if __name__ == "__main__":
    main()
