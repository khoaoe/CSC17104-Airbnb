from __future__ import annotations
from typing import Tuple, Dict, Iterable
import numpy as np


# --- Chuẩn hoá Z-score ---
def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X - mean)/std theo cột; trả về (X_std, mean, std)"""
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0, ddof=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


# --- Chỉ số lỗi cơ bản ---
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_pred.ravel() - y_true.ravel()
    return float(np.sqrt(np.mean(e * e)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = np.abs(y_pred.ravel() - y_true.ravel())
    return float(np.mean(e))


# ============== MODELING PIPELINES ==============
# --- 1) Chia train/val/test (stratified, có seed) ---
# -- 1.1) Tạo nhãn stratify từ y --
def make_stratify_labels(
    y: np.ndarray,
    *,
    task: str = "auto",
    n_bins: int = 10,
    strategy: str = "quantile",  # 'quantile' | 'uniform'
) -> np.ndarray:
    """
    Trả về mảng nhãn để stratify.
    - task='classification': dùng y (cast sang int/str).
    - task='regression'   : băm y thành bin rồi dùng bin-id.
    - task='auto'         : nếu số unique của y <= 20 -> coi như classification, ngược lại coi là regression.
    """
    y = np.asarray(y)
    # auto nhận dạng
    if task == "auto":
        uniq = np.unique(y)
        if uniq.size <= 20 and np.all(np.equal(np.mod(uniq, 1), 0)):  # nhiều khả năng là nhãn rời rạc
            task = "classification"
        else:
            task = "regression"

    # classification -> trả nhãn trực tiếp (string cho chắc)
    if task == "classification":
        return y.astype(str)

    # regression -> tạo bin theo chiến lược
    y_float = y.astype(float)
    # rơi vào NaN -> tạm thay bằng median để không rớt bin
    if np.isnan(y_float).any():
        med = float(np.nanmedian(y_float))
        y_float = np.where(np.isnan(y_float), med, y_float)

    if strategy == "quantile":
        # biên theo quantile, tránh trùng lặp bằng unique
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(y_float, qs))
    else:
        y_min, y_max = float(np.min(y_float)), float(np.max(y_float))
        edges = np.linspace(y_min, y_max, n_bins + 1)

    # nếu edges quá ít (dữ liệu trùng), giảm số bin tối thiểu còn 2
    if edges.size <= 2:
        edges = np.array([np.min(y_float), np.max(y_float)], dtype=float)
        edges = np.unique(edges)
        if edges.size == 1:  # mọi giá trị như nhau
            return np.zeros_like(y_float, dtype=int)

    # digitize -> bin-id trong [0..n_bins-1]
    # đảm bảo max rơi vào bin cuối
    bins = np.digitize(y_float, edges[1:-1], right=False)
    return bins.astype(int)

# -- 1.2) Stratified train/val/test split --
def stratified_train_val_test_idx(
    y: np.ndarray,
    *,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
    task: str = "auto",
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chia index theo stratify (giữ phân bố nhãn/bin gần giống ở mỗi tập).
    - y: vector mục tiêu (có thể rời rạc hoặc liên tục).
    - task: 'classification' | 'regression' | 'auto'
    - Với regression: băm y thành n_bins theo 'quantile' (mặc định).
    """
    rng = np.random.default_rng(seed)
    labels = make_stratify_labels(y, task=task, n_bins=n_bins, strategy=strategy).astype(str)

    idx_train, idx_val, idx_test = [], [], []
    for lab in np.unique(labels):
        lab_idx = np.where(labels == lab)[0]
        rng.shuffle(lab_idx)

        n = lab_idx.size
        n_test = int(n * test_size)
        n_val  = int(n * val_size)

        test_idx = lab_idx[:n_test]
        val_idx  = lab_idx[n_test:n_test + n_val]
        train_idx = lab_idx[n_test + n_val:]

        if test_idx.size: idx_test.append(test_idx)
        if val_idx.size:  idx_val.append(val_idx)
        if train_idx.size: idx_train.append(train_idx)

    if len(idx_train) == 0:
        raise ValueError("Không thể stratify: mỗi nhóm quá nhỏ. Giảm n_bins hoặc thay đổi val_size/test_size.")

    return (
        np.concatenate(idx_train, axis=0),
        np.concatenate(idx_val, axis=0) if len(idx_val) else np.array([], dtype=int),
        np.concatenate(idx_test, axis=0) if len(idx_test) else np.array([], dtype=int),
    )

# -- 1.3) Stratified K-Fold (classification hoặc regression qua bin) --
def stratified_kfold_indices(
    y: np.ndarray,
    *,
    k: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    task: str = "auto",
    n_bins: int = 10,
    strategy: str = "quantile",
):
    """
    Sinh (train_idx, val_idx) cho K-Fold theo phân tầng.
    - Với regression: dùng bin của y để giữ phân bố gần giống nhau giữa các fold.
    """
    rng = np.random.default_rng(seed)
    labels = make_stratify_labels(y, task=task, n_bins=n_bins, strategy=strategy).astype(str)

    # gom index theo nhãn/bin
    buckets = []
    for lab in np.unique(labels):
        lab_idx = np.where(labels == lab)[0]
        if shuffle:
            rng.shuffle(lab_idx)
        # chia lab_idx thành k phần gần bằng nhau
        parts = np.array_split(lab_idx, k)
        buckets.append(parts)

    for i in range(k):
        val_parts = [b[i] for b in buckets if b[i].size]
        val_idx = np.concatenate(val_parts, axis=0) if val_parts else np.array([], dtype=int)

        train_parts = []
        for b in buckets:
            for j in range(k):
                if j != i and b[j].size:
                    train_parts.append(b[j])
        train_idx = np.concatenate(train_parts, axis=0) if train_parts else np.array([], dtype=int)

        if val_idx.size == 0 or train_idx.size == 0:
            raise ValueError("Fold rỗng do nhóm quá nhỏ. Giảm k hoặc giảm n_bins.")
        yield train_idx, val_idx

# -- non-stratified train/val/test split (random, có seed) --
def split_train_val_test_idx(
    n: int,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trả về (idx_train, idx_val, idx_test).
    - Không stratify để giữ đơn giản; với hồi quy, stratify bằng quantile có thể thêm sau
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(test_size * n))
    n_val = int(round(val_size * n))
    idx_test = idx[:n_test]
    idx_val  = idx[n_test:n_test + n_val]
    idx_train = idx[n_test + n_val:]
    return idx_train, idx_val, idx_test


# --- 2) Thêm cột bias tiện lợi ---
def add_bias(X: np.ndarray) -> np.ndarray:
    """Thêm cột 1.0 ở đầu ma trận đặc trưng"""
    return np.concatenate([np.ones((X.shape[0], 1), dtype=float), X.astype(float)], axis=1)


# --- 3) Pipeline Linear/Ridge đơn giản (chuẩn hoá + bias + ridge) ---
def linreg_fit(X: np.ndarray, y: np.ndarray, ridge_alpha: float = 1e-6) -> np.ndarray:
    """
    Nghiệm thường (X^T X + alpha I)^(-1) X^T y.
    - Thêm 1 cột bias bên ngoài trước khi gọi nếu muốn intercept
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n_feat = X.shape[1]
    A = X.T @ X + ridge_alpha * np.eye(n_feat)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w


def linreg_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Dự đoán y = X @ w"""
    return X @ w


def fit_linear_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1e-6,
    use_standardize: bool = True,
    use_bias: bool = True,
    target_transform=None,   # e.g., np.log1p
    target_inverse=None,     # e.g., np.expm1
) -> Dict[str, object]:
    """
    Huấn luyện tuyến tính (Ridge) với lựa chọn biến đổi mục tiêu.
    - Nếu target_transform != None: mô hình fit trên t = f(y).
    """
    X = X.astype(float)
    y = y.astype(float).ravel()
    if target_transform is not None:
        t = target_transform(y)
    else:
        t = y

    if use_standardize:
        Xs, mean, std = standardize(X)
    else:
        Xs, mean, std = X, np.zeros(X.shape[1], float), np.ones(X.shape[1], float)

    Xf = add_bias(Xs) if use_bias else Xs
    w = linreg_fit(Xf, t, ridge_alpha=alpha)

    return {
        "alpha": float(alpha),
        "use_standardize": bool(use_standardize),
        "use_bias": bool(use_bias),
        "w": w,
        "mean": mean,
        "std": std,
        "n_features": X.shape[1],
        "target_transform": target_transform,
        "target_inverse": target_inverse,
    }

def predict_linear_pipeline(X: np.ndarray, model: Dict[str, object]) -> np.ndarray:
    """Dự đoán: nếu có target_inverse -> trả về trên thang gốc."""
    X = X.astype(float)
    if model.get("use_standardize", True):
        mean = model["mean"]; std = model["std"].copy()
        std[std == 0] = 1.0
        X = (X - mean) / std
    if model.get("use_bias", True):
        X = add_bias(X)
    yhat_t = linreg_predict(X, model["w"])
    inv = model.get("target_inverse", None)
    return inv(yhat_t) if inv is not None else yhat_t


# --- 4) Đánh giá chỉ số hồi quy gọn ---
def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 = 1 - SSE/SST (có thể âm nếu mô hình tệ hơn baseline mean)
    """
    y = y_true.ravel().astype(float)
    p = y_pred.ravel().astype(float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # nếu y hằng, quy ước R^2 = 0.0
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Gom metrics: RMSE, MAE, R2"""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score_np(y_true, y_pred),
    }


# --- 5) Baseline: dự đoán trung bình train ---
def baseline_predict_mean(y_train: np.ndarray, n_pred: int) -> np.ndarray:
    """Dự đoán hằng = mean(y_train)"""
    c = float(np.mean(y_train.astype(float)))
    return np.full(n_pred, c, dtype=float)


# --- 6) K-Fold CV cho Ridge: chọn alpha tối ưu theo RMSE ---
def kfold_indices(n: int, k: int = 5, shuffle: bool = True, seed: int = 42):
    """Sinh (train_idx, val_idx) cho KFold đơn giản"""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i], axis=0)
        yield train_idx, val_idx


def cv_ridge_alphas(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Iterable[float],
    *,
    k: int = 5,
    use_standardize: bool = True,
    use_bias: bool = True,
    seed: int = 42,
    # mới:
    use_stratified: bool = False,
    stratify_task: str = "regression",
    stratify_n_bins: int = 10,
    stratify_strategy: str = "quantile",
    target_transform=None,
    target_inverse=None,
) -> Dict[str, object]:
    """
    K-Fold CV để chọn alpha cho Ridge
    - Nếu truyền target_transform/target_inverse: CV tính RMSE/MAE/R2 trên thang gốc
    - Có tuỳ chọn stratified split cho hồi quy qua bin (ổn định hơn khi y lệch).
    """
    X = X.astype(float); y = y.astype(float).ravel()
    scores = []

    # chọn generator fold
    if use_stratified:
        from .models import stratified_kfold_indices  # đã có ở bước trước
        fold_gen = stratified_kfold_indices(y, k=k, shuffle=True, seed=seed,
                                            task=stratify_task, n_bins=stratify_n_bins,
                                            strategy=stratify_strategy)
        folds = list(fold_gen)
    else:
        from .models import kfold_indices
        folds = list(kfold_indices(len(y), k=k, shuffle=True, seed=seed))

    for a in alphas:
        rmse_list, mae_list, r2_list = [], [], []
        for tr, va in folds:
            model = fit_linear_pipeline(
                X[tr], y[tr],
                alpha=float(a),
                use_standardize=use_standardize, use_bias=use_bias,
                target_transform=target_transform, target_inverse=target_inverse,
            )
            pred = predict_linear_pipeline(X[va], model)  # trả về thang gốc nếu inverse != None
            m = evaluate_regression(y[va], pred)
            rmse_list.append(m["rmse"]); mae_list.append(m["mae"]); r2_list.append(m["r2"])
        scores.append({
            "alpha": float(a),
            "rmse_mean": float(np.mean(rmse_list)),
            "mae_mean": float(np.mean(mae_list)),
            "r2_mean": float(np.mean(r2_list)),
        })
    order = np.argsort([s["rmse_mean"] for s in scores])
    best = scores[int(order[0])] if len(scores) else None
    return {"scores": scores, "best": best}
