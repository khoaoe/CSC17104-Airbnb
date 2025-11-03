from typing import Iterator, Tuple

import numpy as np


class StandardScaler:
    """Chuan hoa z-score thuong dung."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            msg = "StandardScaler must be fitted before transform."
            raise RuntimeError(msg)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LinearReg:
    """Linear regression don gian: nghiem dong hoac gradient descent."""

    def __init__(self, l2: float = 0.0, lr: float | None = None, epochs: int = 0) -> None:
        self.l2 = float(l2)
        self.lr = lr
        self.epochs = int(epochs)
        self.w_: np.ndarray | None = None

    def fit_normal(self, X: np.ndarray, y: np.ndarray) -> "LinearReg":
        n, d = X.shape
        del n  # not used in closed form
        A = X.T @ X + self.l2 * np.eye(d)
        b = X.T @ y
        self.w_ = np.linalg.solve(A, b)
        return self

    def fit_gd(self, X: np.ndarray, y: np.ndarray) -> "LinearReg":
        n, d = X.shape
        lr = self.lr or 1e-2
        w = np.zeros(d, dtype=np.float64)
        for _ in range(self.epochs or 1000):
            r = X @ w - y
            grad = (X.T @ r) / n + self.l2 * w
            w -= lr * grad
        self.w_ = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            msg = "LinearReg must be fitted before predict."
            raise RuntimeError(msg)
        return X @ self.w_


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err * err)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


class KFold:
    """Chia K-fold thuong dung, ho tro shuffle."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: int = 42) -> None:
        self.k = int(n_splits)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(idx)
        folds = np.array_split(idx, self.k)
        for i in range(self.k):
            val_idx = folds[i]
            if self.k > 1:
                train_idx = np.concatenate([folds[j] for j in range(self.k) if j != i])
            else:
                train_idx = idx
            yield train_idx, val_idx
