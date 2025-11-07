# Trực quan hoá với Matplotlib (và dùng Seaborn nếu có)
from __future__ import annotations
from typing import List
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


# --- Histogram cho cột số ---
def plot_hist(x: np.ndarray, bins: int = 30, log: bool = False, title: str = "", xlabel: str = "", ylabel: str = "Count"):
    """Histogram nhanh; bỏ NaN trước khi vẽ"""
    x = x.astype(float)
    x = x[~np.isnan(x)]
    plt.figure()
    plt.hist(x, bins=bins, log=log)
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


# --- Bar chart tần suất cho cột phân loại (có top-k) ---
def plot_bar_counts(categories: np.ndarray, topk: int = 10, title: str = "", rotation: int = 0):
    """Vẽ bar chart cho top-k giá trị xuất hiện nhiều nhất"""
    vals, counts = np.unique(categories.astype(str), return_counts=True)
    order = np.argsort(-counts)[:topk]
    vals, counts = vals[order], counts[order]
    plt.figure()
    plt.bar(vals, counts)
    if title: plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()


# --- Scatter geo: lat/lon, có thể tô màu theo giá ---
def plot_scatter_geo(lat: np.ndarray, lon: np.ndarray, c: np.ndarray | None = None, s: int = 6, title: str = ""):
    """Scatter (lon, lat); nếu có c -> dùng như màu (ví dụ: price)"""
    x = lon.astype(float)
    y = lat.astype(float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    plt.figure()
    sc = plt.scatter(x, y, s=s, c=c[m] if (c is not None) else None, alpha=0.6)
    if c is not None:
        plt.colorbar(sc, label="Value")
    if title: plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()


# --- Heatmap ma trận tương quan ---
def plot_corr_heatmap(C: np.ndarray, labels: List[str], title: str = "Correlation"):
    """Vẽ heatmap corr; ưu tiên seaborn nếu có"""
    plt.figure()
    if _HAS_SNS:
        print("Using seaborn for heatmap visualization")
        sns.heatmap(C, xticklabels=labels, yticklabels=labels, annot=False, square=True)
    else:
        im = plt.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()


# --- Boxplot số theo nhóm (ví dụ: price theo room_type) ---
def plot_box_by_cat(values: np.ndarray, categories: np.ndarray, title: str = "", ylabel: str = ""):
    """Boxplot value theo từng nhóm từ 'categories'"""
    vals = values.astype(float)
    cats = categories.astype(str)
    u = np.unique(cats)
    data = [vals[cats == g] for g in u]
    plt.figure()
    plt.boxplot(data, labels=u, showfliers=False)
    if title: plt.title(title)
    if ylabel: plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.tight_layout()
    
# --- Histogram với vạch phân vị ---
def plot_hist_with_quantiles(x: np.ndarray, qs=(1, 50, 99), bins: int = 50, title: str = "", xlabel: str = ""):
    """Histogram + vạch phân vị để quan sát ngoại lai nhanh"""
    x = x.astype(float)
    x = x[~np.isnan(x)]
    plt.figure()
    plt.hist(x, bins=bins)
    if qs:
        qv = np.percentile(x, qs)
        for v in qv:
            plt.axvline(v, linestyle="--")
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()


# --- Plot cho hồi quy ---
def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Pred vs True"):
    """Scatter y_true vs y_pred + đường y=x."""
    y = y_true.astype(float).ravel()
    p = y_pred.astype(float).ravel()
    plt.figure()
    plt.scatter(y, p, s=10, alpha=0.6)
    lims = [min(np.min(y), np.min(p)), max(np.max(y), np.max(p))]
    plt.plot(lims, lims, "--")
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(title)
    plt.tight_layout()

def plot_residuals_hist(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 50, title: str = "Residuals"):
    """Histogram residual (y - yhat)."""
    e = y_true.astype(float).ravel() - y_pred.astype(float).ravel()
    plt.figure()
    plt.hist(e, bins=bins)
    plt.title(title); plt.xlabel("Residual"); plt.ylabel("Count")
    plt.tight_layout()
