from __future__ import annotations
from typing import Tuple
import numpy as np


# --- Chia train/test theo index ---
def train_test_split_idx(n: int, test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Trả về (idx_train, idx_test)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(test_size * n))
    return idx[n_test:], idx[:n_test]


# --- Chuẩn hoá Z-score ---
def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X - mean)/std theo cột; trả về (X_std, mean, std)."""
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0, ddof=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


# --- Hồi quy tuyến tính (closed-form) với Ridge nhỏ chống suy biến ---
def linreg_fit(X: np.ndarray, y: np.ndarray, ridge_alpha: float = 1e-6) -> np.ndarray:
    """
    Nghiệm thường (X^T X + alpha I)^(-1) X^T y.
    - Thêm 1 cột bias bên ngoài trước khi gọi nếu muốn intercept.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n_feat = X.shape[1]
    A = X.T @ X + ridge_alpha * np.eye(n_feat)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w


def linreg_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Dự đoán y = X @ w."""
    return X @ w


# --- Chỉ số lỗi cơ bản ---
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_pred.ravel() - y_true.ravel()
    return float(np.sqrt(np.mean(e * e)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = np.abs(y_pred.ravel() - y_true.ravel())
    return float(np.mean(e))
