"""
module_a_outlier_detection.py
Module A: Dynamic Outlier Detection & Cost-Sensitive Anomaly Screening.
Combines:
1. Mahalanobis Distance (Covariance-aware multi-feature statistical distance)
2. Isolation Forest (Tree-based partitioning for contextual isolation)
3. Local Outlier Factor (LOF - Density-based local neighborhood anomaly scoring)
4. Ensemble Hybrid Scorer
5. F_beta Threshold Optimizer (Beta >= 2, zero escape / high recall target)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


class MahalanobisDetector:
    """
    Statistical anomaly detector using Regularized Mahalanobis Distance.
    Factors in covariance between multiple physical sensor channels.
    """

    def __init__(self, regularize: bool = True):
        self.regularize = regularize
        self.mean_ = None
        self.inv_cov_ = None

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        X = np.asarray(X, dtype=np.float64)
        if self.regularize:
            # Ledoit-Wolf shrinkage ensures non-singular, well-conditioned covariance
            cov_estimator = LedoitWolf().fit(X)
            self.mean_ = cov_estimator.location_
            self.inv_cov_ = cov_estimator.precision_ # precision is the inverse covariance
        else:
            cov_estimator = EmpiricalCovariance().fit(X)
            self.mean_ = cov_estimator.location_
            # Use pseudo-inverse for numerical stability
            self.inv_cov_ = np.linalg.pinv(cov_estimator.covariance_)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates Mahalanobis distance D_M for each sample.
        Higher distance indicates higher anomaly degree.
        """
        X = np.asarray(X, dtype=np.float64)
        diff = X - self.mean_
        # Vectorized Mahalanobis calculation: sqrt(diag(diff * inv_cov * diff^T))
        # Equivalent to: sqrt(sum((diff @ inv_cov) * diff, axis=1))
        dist_sq = np.sum((diff @ self.inv_cov_) * diff, axis=1)
        dist_sq = np.clip(dist_sq, 0, None)
        return np.sqrt(dist_sq)


class DynamicOutlierDetector:
    """
    Integrated Multi-Model Outlier Detector for Semiconductor Burn-In Screening.
    Combines Mahalanobis Distance, Isolation Forest, and Local Outlier Factor (LOF).
    """

    def __init__(
        self,
        contamination: float = 0.06,
        n_estimators: int = 150,
        n_neighbors: int = 25,
        weights: Optional[Dict[str, float]] = None,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.weights = weights or {"mahalanobis": 0.35, "isolation_forest": 0.40, "lof": 0.25}
        
        self.scaler = RobustScaler()
        self.mahalanobis_model = MahalanobisDetector(regularize=True)
        self.iso_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True, # enables score_samples on new test data
            n_jobs=-1
        )
        
        self.score_min_max_ = {}
        self.optimal_threshold_ = 0.5

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "DynamicOutlierDetector":
        """
        Fits Mahalanobis, Isolation Forest, and LOF on baseline feature space.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X)

        # Scale features using RobustScaler (outlier-resilient median/IQR scaling)
        X_scaled = self.scaler.fit_transform(X_arr)
        
        # 1. Fit Mahalanobis
        self.mahalanobis_model.fit(X_scaled)
        m_raw = self.mahalanobis_model.score_samples(X_scaled)
        
        # 2. Fit Isolation Forest
        self.iso_forest.fit(X_scaled)
        # Isolation forest score: lower score means more anomalous -> invert so higher = more anomalous
        if_raw = -self.iso_forest.score_samples(X_scaled)
        
        # 3. Fit LOF
        self.lof.fit(X_scaled)
        lof_raw = -self.lof.score_samples(X_scaled)
        
        # Store min/max bounds for min-max normalization to [0, 1]
        self.score_min_max_ = {
            "mahalanobis": (np.percentile(m_raw, 1), np.percentile(m_raw, 99)),
            "isolation_forest": (np.percentile(if_raw, 1), np.percentile(if_raw, 99)),
            "lof": (np.percentile(lof_raw, 1), np.percentile(lof_raw, 99)),
        }
        return self

    def _normalize_score(self, scores: np.ndarray, key: str) -> np.ndarray:
        s_min, s_max = self.score_min_max_[key]
        if s_max <= s_min:
            return np.zeros_like(scores)
        norm = (scores - s_min) / (s_max - s_min)
        return np.clip(norm, 0.0, 1.0)

    def predict_anomaly_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Calculates individual and ensemble normalized anomaly scores for all units.
        Returns a DataFrame containing:
        - mahalanobis_score [0-1]
        - iforest_score [0-1]
        - lof_score [0-1]
        - ensemble_score [0-1]
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        X_scaled = self.scaler.transform(X_arr)
        
        m_raw = self.mahalanobis_model.score_samples(X_scaled)
        if_raw = -self.iso_forest.score_samples(X_scaled)
        lof_raw = -self.lof.score_samples(X_scaled)
        
        m_norm = self._normalize_score(m_raw, "mahalanobis")
        if_norm = self._normalize_score(if_raw, "isolation_forest")
        lof_norm = self._normalize_score(lof_raw, "lof")
        
        # Weighted ensemble
        w = self.weights
        w_sum = sum(w.values())
        ensemble = (
            w["mahalanobis"] * m_norm +
            w["isolation_forest"] * if_norm +
            w["lof"] * lof_norm
        ) / w_sum
        
        return pd.DataFrame({
            "mahalanobis_score": m_norm,
            "iforest_score": if_norm,
            "lof_score": lof_norm,
            "ensemble_anomaly_score": ensemble
        })

    def optimize_fbeta_threshold(
        self,
        y_true: np.ndarray,
        anomaly_scores: np.ndarray,
        beta: float = 2.0,
        n_thresholds: int = 200
    ) -> Dict[str, Union[float, Dict]]:
        """
        Finds the decision threshold that maximizes the F_beta score.
        For semiconductor screening, beta=2.0 or 3.0 places high priority
        on Recall (avoiding escapes/False Negatives).
        
        F_beta = (1 + beta^2) * (Precision * Recall) / ((beta^2 * Precision) + Recall)
        """
        thresholds = np.linspace(0.05, 0.95, n_thresholds)
        best_threshold = 0.5
        best_fbeta = -1.0
        best_metrics = {}
        
        metrics_history = []
        
        for thresh in thresholds:
            y_pred = (anomaly_scores >= thresh).astype(int)
            
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            escape_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0 # False Negative Rate
            
            beta_sq = beta ** 2
            if (beta_sq * precision + recall) > 0:
                fbeta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
            else:
                fbeta = 0.0
                
            metrics_history.append({
                "threshold": thresh,
                "fbeta": fbeta,
                "precision": precision,
                "recall": recall,
                "escape_rate": escape_rate,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            })
            
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_threshold = thresh
                best_metrics = {
                    "threshold": thresh,
                    "fbeta": fbeta,
                    "precision": precision,
                    "recall": recall,
                    "escape_rate": escape_rate,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn
                }

        self.optimal_threshold_ = best_threshold
        return {
            "optimal_threshold": best_threshold,
            "best_fbeta": best_fbeta,
            "beta": beta,
            "best_metrics": best_metrics,
            "history": pd.DataFrame(metrics_history)
        }

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predicts binary anomaly flag (1 = Outlier/Reject, 0 = In-Spec/Pass).
        Uses calibrated optimal threshold if threshold is not provided.
        """
        thresh = threshold if threshold is not None else self.optimal_threshold_
        scores_df = self.predict_anomaly_scores(X)
        return (scores_df["ensemble_anomaly_score"].values >= thresh).astype(int)


class DPATEngine:
    """
    Dynamic Part Average Testing (DPAT) Engine compliant with AEC-Q001 standard.
    
    Static limits fail to catch latent defects that operate within absolute datasheet specs
    but deviate drastically from their production lot distribution.
    
    Example:
      If a lot has an average leakage current of 10 uA (sigma = 2.5 uA),
      a component showing 45 uA is an extreme +14.0 sigma outlier,
      even though it falls beneath an absolute datasheet maximum limit of 50 uA.
    """

    def __init__(self, k_sigma: float = 3.0, robust: bool = True):
        """
        Args:
            k_sigma: Multiplier for standard deviation / pseudo-sigma (default 3.0 or 4.0).
            robust: If True, uses Median and IQR (pseudo-sigma = IQR * 0.7413) to resist
                    outlier contamination in baseline estimation.
        """
        self.k_sigma = k_sigma
        self.robust = robust
        self.lot_limits_ = {}

    def fit_lot_limits(
        self,
        df: pd.DataFrame,
        param_col: str,
        lot_col: str = "lot_id"
    ) -> "DPATEngine":
        """
        Calculates dynamic screening limits for each lot.
        Upper DPAT Limit = Center + k * Spread
        Lower DPAT Limit = Center - k * Spread
        """
        self.lot_limits_[param_col] = {}
        for lot_id, group in df.groupby(lot_col):
            vals = group[param_col].dropna().values
            if len(vals) == 0:
                continue
            if self.robust:
                median = float(np.median(vals))
                q75, q25 = np.percentile(vals, [75, 25])
                iqr = q75 - q25
                pseudo_sigma = max(iqr * 0.7413, 1e-6)
                center = median
                spread = pseudo_sigma
            else:
                center = float(np.mean(vals))
                spread = float(np.std(vals))
                if spread < 1e-6:
                    spread = 1e-6

            upper_limit = center + self.k_sigma * spread
            lower_limit = center - self.k_sigma * spread
            self.lot_limits_[param_col][lot_id] = {
                "center": center,
                "spread": spread,
                "upper_limit": upper_limit,
                "lower_limit": lower_limit,
            }
        return self

    def screen_units(
        self,
        df: pd.DataFrame,
        param_col: str,
        static_max_limit: Optional[float] = None,
        lot_col: str = "lot_id"
    ) -> pd.DataFrame:
        """
        Screens units using dynamic DPAT limits and compares against static specification.
        Returns a DataFrame with DPAT bounds, flags, and latent defect escape status.
        """
        out_df = df.copy()
        if param_col not in self.lot_limits_:
            self.fit_lot_limits(df, param_col, lot_col)

        lot_dict = self.lot_limits_[param_col]
        upper_limits = []
        lower_limits = []
        z_scores = []
        dpat_rejects = []
        static_rejects = []
        latent_escapes = []

        for idx, row in df.iterrows():
            lot_id = row.get(lot_col, "LOT_DEFAULT")
            val = float(row[param_col])
            limits = lot_dict.get(lot_id, None)

            if limits is None:
                center = df[param_col].mean()
                spread = max(df[param_col].std(), 1e-6)
                u_lim = center + self.k_sigma * spread
                l_lim = center - self.k_sigma * spread
            else:
                center = limits["center"]
                spread = limits["spread"]
                u_lim = limits["upper_limit"]
                l_lim = limits["lower_limit"]

            z = (val - center) / spread
            is_dpat_reject = bool(val > u_lim or val < l_lim)
            is_static_reject = bool(val > static_max_limit) if static_max_limit is not None else False

            # Latent defect escape: Passes static spec but fails dynamic DPAT
            is_latent_escape = bool((not is_static_reject) and is_dpat_reject)

            upper_limits.append(u_lim)
            lower_limits.append(l_lim)
            z_scores.append(z)
            dpat_rejects.append(int(is_dpat_reject))
            static_rejects.append(int(is_static_reject))
            latent_escapes.append(int(is_latent_escape))

        out_df[f"{param_col}_dpat_upper"] = upper_limits
        out_df[f"{param_col}_dpat_lower"] = lower_limits
        out_df[f"{param_col}_zscore"] = z_scores
        out_df[f"{param_col}_dpat_reject"] = dpat_rejects
        out_df[f"{param_col}_static_reject"] = static_rejects
        out_df[f"{param_col}_latent_defect_escape"] = latent_escapes
        return out_df

    @staticmethod
    def evaluate_latent_defect_benchmark(
        lot_mean: float = 10.0,
        lot_std: float = 2.5,
        candidate_value: float = 45.0,
        static_spec_limit: float = 50.0,
        k_sigma: float = 3.0
    ) -> Dict[str, Union[float, str, bool]]:
        """
        Direct mathematical demonstration of the ISRO benchmark case:
        Lot mean = 10 uA, Part = 45 uA, Spec Limit = 50 uA.
        """
        dpat_upper = lot_mean + k_sigma * lot_std
        z_score = (candidate_value - lot_mean) / lot_std
        passes_static = candidate_value <= static_spec_limit
        passes_dpat = candidate_value <= dpat_upper
        latent_defect_escaped = passes_static and (not passes_dpat)

        return {
            "lot_mean_ua": lot_mean,
            "lot_std_ua": lot_std,
            "candidate_value_ua": candidate_value,
            "static_spec_limit_ua": static_spec_limit,
            "dpat_upper_limit_ua": round(dpat_upper, 3),
            "z_score_sigma": round(z_score, 2),
            "static_screening_disposition": "PASS (ESCAPE!)" if passes_static else "REJECT",
            "dpat_dynamic_disposition": "REJECT (CAUGHT)" if not passes_dpat else "PASS",
            "is_latent_defect": True,
            "escapes_static_screening": latent_defect_escaped,
            "physics_disposition": (
                f"Static test PASSES ({candidate_value}uA <= {static_spec_limit}uA) allowing field escape! "
                f"DPAT DYNAMICALLY REJECTS ({candidate_value}uA > {dpat_upper:.1f}uA, Z={z_score:+.1f}sigma). "
                f"Prevents mission-critical satellite infant mortality."
            )
        }


class GDBNSpatialEngine:
    """
    Good Die in Bad Neighborhood (GDBN) Spatial Risk Engine.
    Evaluates spatial defect clustering on wafer or burn-in carrier socket maps.
    Dies adjacent to multiple failing components inherit high latent defect risk.
    """

    def __init__(self, neighborhood_radius: float = 1.5, defect_density_threshold: float = 0.35):
        self.neighborhood_radius = neighborhood_radius
        self.defect_density_threshold = defect_density_threshold

    def compute_spatial_risk(
        self,
        df: pd.DataFrame,
        x_col: str = "socket_x",
        y_col: str = "socket_y",
        defect_flag_col: str = "dpat_reject"
    ) -> pd.DataFrame:
        """
        Calculates neighbor defect count and spatial defect density for each component.
        """
        out_df = df.copy()
        coords = out_df[[x_col, y_col]].values
        flags = out_df[defect_flag_col].values
        n = len(df)

        neighbor_defects = np.zeros(n, dtype=int)
        neighbor_totals = np.zeros(n, dtype=int)
        spatial_risks = np.zeros(n, dtype=float)
        gdbn_rejects = np.zeros(n, dtype=int)

        for i in range(n):
            xi, yi = coords[i]
            dists = np.sqrt((coords[:, 0] - xi) ** 2 + (coords[:, 1] - yi) ** 2)
            # Find neighbors within radius, excluding self
            mask = (dists > 0) & (dists <= self.neighborhood_radius)
            total_neighbors = int(np.sum(mask))
            if total_neighbors > 0:
                defects_around = int(np.sum(flags[mask]))
                density = defects_around / total_neighbors
            else:
                defects_around = 0
                density = 0.0

            neighbor_defects[i] = defects_around
            neighbor_totals[i] = total_neighbors
            spatial_risks[i] = density
            # GDBN triggers if die itself passed, but neighborhood is heavily contaminated
            gdbn_rejects[i] = int((flags[i] == 0) and (density >= self.defect_density_threshold))

        out_df["gdbn_neighbor_defects"] = neighbor_defects
        out_df["gdbn_neighbor_total"] = neighbor_totals
        out_df["gdbn_spatial_risk_density"] = spatial_risks
        out_df["gdbn_spatial_reject"] = gdbn_rejects
        return out_df

