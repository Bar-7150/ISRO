"""
module_b_drift_predictor.py
Module B: Time-Series Parametric Drift Predictor.
Predicts Value_168h using only Value_0h, Value_24h, Delta_0-24h, and Lot Context:
1. Gradient Boosting Regressor (XGBoost / HistGradientBoosting with MAE / L1 Loss)
2. Bayesian Ridge Regression (Predictive Mean + Variance for Confidence Intervals)
3. Upper Confidence Limit (UCL) Safety Limit Screening for QA Abort
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

# Try importing XGBoost, fallback smoothly to HistGradientBoostingRegressor if needed
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import HistGradientBoostingRegressor


class BurnInDriftForecaster:
    """
    Time-Series Parametric Degradation Predictor for semiconductor burn-in screening.
    
    Predicts late-stage burn-in drift (168h qualification target) from early-life
    measurements (0h, 24h, and Delta_0-24h).
    """

    def __init__(
        self,
        use_xgboost: bool = True,
        xgb_params: Optional[Dict] = None,
        safety_limit_168h: float = 25.0, # e.g. 25 uA maximum permissible leakage at 168h
        confidence_z: float = 1.96, # 95% two-sided confidence bound
        random_state: int = 42
    ):
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self.xgb_params = xgb_params or {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "objective": "reg:absoluteerror", # L1 / MAE loss function
            "random_state": random_state
        }
        self.safety_limit_168h = safety_limit_168h
        self.confidence_z = confidence_z
        self.random_state = random_state

        # Gradient Boosting Model with L1 / MAE loss
        if self.use_xgboost:
            self.gbr_model = xgb.XGBRegressor(**self.xgb_params)
        else:
            self.gbr_model = HistGradientBoostingRegressor(
                loss="absolute_error", # MAE / L1 objective
                max_iter=150,
                max_depth=5,
                learning_rate=0.05,
                random_state=random_state
            )

        # Bayesian Ridge Model for uncertainty quantification
        self.scaler = StandardScaler()
        self.bayesian_model = BayesianRidge(
            max_iter=300,
            compute_score=True
        )
        self.feature_names_ = []
        self.loss_history_ = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple]] = None
    ) -> "BurnInDriftForecaster":
        """
        Fits both the Gradient Boosting Regressor (MAE loss) and Bayesian Ridge model.
        Tracks iteration-by-iteration training and validation loss if eval_set is provided.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values
        else:
            self.feature_names_ = [f"f_{i}" for i in range(X.shape[1])]
            X_mat = np.asarray(X)

        y_vec = np.asarray(y).ravel()

        # 1. Fit Gradient Boosting with MAE & Track Loss
        if self.use_xgboost and eval_set:
            eval_pairs = [(X_mat, y_vec)]
            for val_x, val_y in eval_set:
                vx = val_x.values if isinstance(val_x, pd.DataFrame) else np.asarray(val_x)
                vy = np.asarray(val_y).ravel()
                eval_pairs.append((vx, vy))
            
            self.gbr_model.fit(X_mat, y_vec, eval_set=eval_pairs, verbose=False)
            eval_res = self.gbr_model.evals_result()
            metric_key = list(eval_res["validation_0"].keys())[0]
            self.loss_history_["train_loss"] = eval_res["validation_0"][metric_key]
            if len(eval_pairs) > 1:
                self.loss_history_["val_loss"] = eval_res["validation_1"][metric_key]
        elif self.use_xgboost:
            self.gbr_model.fit(X_mat, y_vec, eval_set=[(X_mat, y_vec)], verbose=False)
            eval_res = self.gbr_model.evals_result()
            metric_key = list(eval_res["validation_0"].keys())[0]
            self.loss_history_["train_loss"] = eval_res["validation_0"][metric_key]
        else:
            self.gbr_model.fit(X_mat, y_vec)
            if hasattr(self.gbr_model, "train_score_"):
                self.loss_history_["train_loss"] = list(self.gbr_model.train_score_)

        # 2. Fit Bayesian Ridge on scaled features
        X_scaled = self.scaler.fit_transform(X_mat)
        self.bayesian_model.fit(X_scaled, y_vec)

        return self

    def get_loss_history(self) -> Dict[str, List[float]]:
        """Returns iteration-by-iteration MAE loss history."""
        return self.loss_history_

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Generates point predictions from Gradient Boosting and
        probabilistic predictions (mean, std, 95% UCL) from Bayesian Ridge.
        
        Returns a DataFrame with columns:
        - pred_168h_gbr (Point prediction via MAE-optimized gradient boosting)
        - pred_168h_bayes_mean (Posterior predictive mean)
        - pred_168h_bayes_std (Predictive uncertainty standard deviation)
        - pred_168h_ucl (Upper Confidence Limit = mean + z * std)
        - ucl_exceeds_limit (Boolean QA flag: 1 if UCL > safety_limit_168h)
        """
        if isinstance(X, pd.DataFrame):
            X_mat = X.values
        else:
            X_mat = np.asarray(X)

        # Gradient Boosting MAE prediction
        pred_gbr = self.gbr_model.predict(X_mat)

        # Bayesian Ridge predictive distribution
        X_scaled = self.scaler.transform(X_mat)
        pred_bayes_mean, pred_bayes_std = self.bayesian_model.predict(
            X_scaled, return_std=True
        )

        # Calculate 95% Upper Confidence Limit (UCL)
        ucl = pred_bayes_mean + (self.confidence_z * pred_bayes_std)
        ucl_flag = (ucl > self.safety_limit_168h).astype(int)

        results = pd.DataFrame({
            "pred_168h_gbr": pred_gbr,
            "pred_168h_bayes_mean": pred_bayes_mean,
            "pred_168h_bayes_std": pred_bayes_std,
            "pred_168h_ucl": ucl,
            "ucl_exceeds_safety_limit": ucl_flag
        })
        return results

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Computes evaluation metrics prioritizing MAE and Median Absolute Error.
        """
        preds_df = self.predict(X)
        y_t = np.asarray(y_true).ravel()
        
        mae_gbr = mean_absolute_error(y_t, preds_df["pred_168h_gbr"])
        medae_gbr = median_absolute_error(y_t, preds_df["pred_168h_gbr"])
        r2_gbr = r2_score(y_t, preds_df["pred_168h_gbr"])

        mae_bayes = mean_absolute_error(y_t, preds_df["pred_168h_bayes_mean"])
        r2_bayes = r2_score(y_t, preds_df["pred_168h_bayes_mean"])

        # Empirical Coverage: Proportion of actual values falling below the 95% UCL
        coverage_prob = np.mean(y_t <= preds_df["pred_168h_ucl"])

        return {
            "gbr_mae": mae_gbr,
            "gbr_medae": medae_gbr,
            "gbr_r2": r2_gbr,
            "bayesian_mae": mae_bayes,
            "bayesian_r2": r2_bayes,
            "bayesian_ucl_coverage_prob": coverage_prob
        }
