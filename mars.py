import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional

np.random.seed(123)

# 1) Generate synthetic nonlinear seasonal time series
n = 360  # 12 months * 30 days as an example length
t = np.arange(n)

# True process: trend + monthly seasonality + slow seasonal regime + threshold nonlinearity
trend = 0.03 * t
monthly = 8 * np.sin(2 * np.pi * t / 30)           # 30-day cycle
season90 = 3 * np.sign(np.sin(2 * np.pi * t / 90)) # regime flips every ~45 days
threshold_boost = np.where(monthly > 6, 4, 0)      # threshold nonlinearity
y_true = 20 + trend + monthly + season90 + threshold_boost
y = y_true + np.random.normal(scale=1.2, size=n)

# 2) Frame as supervised learning
def make_lag_features(series: np.ndarray, max_lag: int) -> pd.DataFrame:
    s = pd.Series(series)
    data = {f"lag_{k}": s.shift(k) for k in range(1, max_lag + 1)}
    return pd.DataFrame(data)

max_lag = 10
df = make_lag_features(y, max_lag)
# Calendar / seasonal features
df["sin_30"] = np.sin(2 * np.pi * np.arange(len(df)) / 30)
df["cos_30"] = np.cos(2 * np.pi * np.arange(len(df)) / 30)
df["sin_90"] = np.sin(2 * np.pi * np.arange(len(df)) / 90)
df["cos_90"] = np.cos(2 * np.pi * np.arange(len(df)) / 90)
df["time_idx"] = np.arange(len(df))
df["target"] = y
df = df.dropna().reset_index(drop=True)

# Chronological split
train_frac = 0.8
split_idx = int(len(df) * train_frac)
X_train = df.drop(columns=["target"]).iloc[:split_idx].to_numpy()
y_train = df["target"].iloc[:split_idx].to_numpy()
X_test = df.drop(columns=["target"]).iloc[split_idx:].to_numpy()
y_test = df["target"].iloc[split_idx:].to_numpy()

feature_names = df.drop(columns=["target"]).columns.tolist()

# 3) Lightweight MARS-like model (hinge basis + linear terms), forward selection + backward pruning
def hinge(vec: np.ndarray, c: float, direction: int) -> np.ndarray:
    # direction: +1 => max(0, x - c), -1 => max(0, c - x)
    if direction == 1:
        return np.maximum(0.0, vec - c)
    else:
        return np.maximum(0.0, c - vec)

@dataclass
class BasisSpec:
    kind: str                 # "hinge" or "linear"
    feat_idx: int
    knot: Optional[float]     # None for linear
    direction: Optional[int]  # +1 / -1 for hinge, None for linear

@dataclass
class MarsLiteTS:
    basis: List[BasisSpec]
    beta: np.ndarray

def design_matrix(X: np.ndarray, basis_list: List[BasisSpec]) -> np.ndarray:
    parts = [np.ones((X.shape[0], 1))]  # intercept
    for b in basis_list:
        col = X[:, b.feat_idx]
        if b.kind == "linear":
            vec = col
        else:
            vec = hinge(col, b.knot, b.direction)
        parts.append(vec.reshape(-1, 1))
    return np.hstack(parts)

def gcv(y_true: np.ndarray, y_hat: np.ndarray, m_terms: int, penalty: float = 2.0) -> float:
    n = len(y_true)
    rss = np.sum((y_true - y_hat) ** 2)
    # Effective number of parameters: 1 (intercept) + m_terms, scaled by penalty
    c_eff = 1 + penalty * m_terms
    denom = max(1e-3, (1 - c_eff / n))
    return rss / (n * denom ** 2)

def fit_mars_lite_ts(X: np.ndarray, y: np.ndarray, max_terms: int = 20, q_knots: int = 15,
                     allow_linear: bool = True, penalty: float = 2.0, min_improve: float = 1e-3) -> MarsLiteTS:
    n, p = X.shape
    # Candidate knots from quantiles for each feature
    quantiles = np.linspace(0.05, 0.95, q_knots)
    cand_knots = [np.quantile(X[:, j], quantiles) for j in range(p)]

    basis_list: List[BasisSpec] = []
    # Intercept-only baseline
    D = design_matrix(X, basis_list)
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    yhat = D @ beta
    best_score = gcv(y, yhat, m_terms=0, penalty=penalty)

    # Forward selection
    for _ in range(max_terms):
        best_add = None
        best_add_score = best_score
        # Try linear terms
        if allow_linear:
            for j in range(p):
                spec = BasisSpec(kind="linear", feat_idx=j, knot=None, direction=None)
                D_try = design_matrix(X, basis_list + [spec])
                beta_try, *_ = np.linalg.lstsq(D_try, y, rcond=None)
                yhat_try = D_try @ beta_try
                score = gcv(y, yhat_try, m_terms=len(basis_list) + 1, penalty=penalty)
                if score < best_add_score - 1e-12:
                    best_add_score = score
                    best_add = (spec, beta_try, yhat_try)
        # Try hinge terms
        for j in range(p):
            for c in cand_knots[j]:
                for d in (1, -1):
                    spec = BasisSpec(kind="hinge", feat_idx=j, knot=float(c), direction=d)
                    D_try = design_matrix(X, basis_list + [spec])
                    beta_try, *_ = np.linalg.lstsq(D_try, y, rcond=None)
                    yhat_try = D_try @ beta_try
                    score = gcv(y, yhat_try, m_terms=len(basis_list) + 1, penalty=penalty)
                    if score < best_add_score - 1e-12:
                        best_add_score = score
                        best_add = (spec, beta_try, yhat_try)
        if best_add is None or (best_score - best_add_score) < min_improve:
            break
        # Accept the addition
        spec, beta, yhat = best_add
        basis_list.append(spec)
        best_score = best_add_score

    # Backward pruning (greedy)
    improved = True
    while improved and len(basis_list) > 0:
        improved = False
        current_best = best_score
        drop_idx = None
        for j in range(len(basis_list)):
            trial = [b for i, b in enumerate(basis_list) if i != j]
            D_try = design_matrix(X, trial)
            beta_try, *_ = np.linalg.lstsq(D_try, y, rcond=None)
            yhat_try = D_try @ beta_try
            score = gcv(y, yhat_try, m_terms=len(trial), penalty=penalty)
            if score < current_best - 1e-12:
                current_best = score
                drop_idx = j
                best_beta_try = beta_try
                best_yhat_try = yhat_try
        if drop_idx is not None:
            basis_list.pop(drop_idx)
            beta = best_beta_try
            yhat = best_yhat_try
            best_score = current_best
            improved = True

    # Final refit
    D_final = design_matrix(X, basis_list)
    beta_final, *_ = np.linalg.lstsq(D_final, y, rcond=None)
    return MarsLiteTS(basis=basis_list, beta=beta_final)

def predict(model: MarsLiteTS, X: np.ndarray) -> np.ndarray:
    D = design_matrix(X, model.basis)
    return D @ model.beta

# Train model
model = fit_mars_lite_ts(X_train, y_train, max_terms=25, q_knots=12, allow_linear=True, penalty=2.0, min_improve=1e-3)

# Evaluate
y_pred = predict(model, X_test)
rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(1e-6, np.abs(y_test)))) * 100)

# 4) Plot
plt.figure(figsize=(12, 6))
plt.plot(y, label="Actual")
plt.axvline(split_idx + max_lag, linestyle="--", label="Train/Test split")
# align preds to the original indexing
start = split_idx + max_lag
x_axis = np.arange(start, start + len(y_pred))
plt.plot(x_axis, y_pred, label="MARS-like forecast")
plt.title(f"MARS-like Time Series Forecast  |  RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()


