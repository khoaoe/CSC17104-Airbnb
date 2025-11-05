<<<<<<< HEAD
﻿from __future__ import annotations
import os
import sys
import json
import subprocess as sp
from pathlib import Path
from typing import Tuple, Dict, Any
=======
﻿import csv
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List
>>>>>>> parent of d1a2c48 (move to ./src)

import numpy as np

# =========================
# Helpers chung
# =========================

ROOT = Path(__file__).resolve().parents[1]  # repo root: .../project/
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def _echo(msg: str) -> None:
    print(f"[data] {msg}")

# =========================
# Kaggle download (portable)
# =========================
def kaggle_download_if_needed(dataset: str, filename: str, out_dir: Path = RAW_DIR) -> Path:
    """
    Tải file từ Kaggle datasets nếu chưa tồn tại (không unzip thủ công).
    - Nếu filename là .zip => dùng --unzip.
    - Nếu là .csv / .csv.gz => KHÔNG --unzip. Kaggle có thể nén nội bộ; cứ tải đúng tên file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        _echo(f"Đã có: {out_path.name} ({out_path.stat().st_size} bytes)")
        return out_path

    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset,
        "-f", filename,
        "-p", str(out_dir),
        "--quiet",
    ]
    # Nếu là .zip thì thêm --unzip
    if filename.lower().endswith(".zip"):
        cmd.append("--unzip")

    _echo(f"Tải từ Kaggle: {dataset} -> {filename}")
    try:
        sp.run(cmd, check=True)
    except FileNotFoundError:
        # fallback: thử ../.venv/bin/kaggle (nếu user cài vào venv)
        venv_kaggle = ROOT / ".venv" / "bin" / "kaggle"
        if venv_kaggle.exists():
            cmd[0] = str(venv_kaggle)
            sp.run(cmd, check=True)
        else:
            raise RuntimeError("Không tìm thấy lệnh 'kaggle'. Hãy cài kaggle CLI hoặc kích hoạt venv.")

    # Sau tải, nếu là zip + --unzip thì file csv nằm trong out_dir; nếu csv thì đã xong.
    if out_path.exists():
        return out_path
    # Nếu Kaggle đổi tên (ví dụ thêm số), tìm gần đúng:
    candidates = list(out_dir.glob(Path(filename).name))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Không tìm thấy file sau khi tải: {filename} trong {out_dir}")

# =========================
# NumPy-only CSV loader
# =========================
def load_csv_genfromtxt(csv_path: Path, delimiter: str = ",", skip_header: int = 1, dtype=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Đọc CSV bằng numpy.genfromtxt (NumPy-only).
    - Trả về (header, data), với header là mảng 1D dtype=object tên cột.
    - Các cột string sẽ có dtype=object; số => float.
    """
    if dtype is None:
        # Cho phép mixed types; dùng None -> genfromtxt tự suy luận, missing -> np.nan
        dtype = None

    with open(csv_path, "r", encoding="utf-8") as f:
        first = f.readline().rstrip("\n\r")
        header = np.array(first.split(delimiter), dtype=object)

    data = np.genfromtxt(
        csv_path,
        delimiter=delimiter,
        skip_header=skip_header,
        dtype=dtype,
        encoding="utf-8",
        autostrip=True
    )
    # Khi chỉ 1 dòng, genfromtxt trả về 1D -> ép thành 2D
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data

# =========================
# In đầy đủ category nếu ít nhóm
# =========================
def print_small_category_full(values: np.ndarray, max_groups: int = 10, title: str = "Categories") -> None:
    """
    In toàn bộ tên nhóm nếu số nhóm <= max_groups (đề yêu cầu).
    """
    uniq, counts = np.unique(values.astype(object), return_counts=True)
    order = np.argsort(-counts)
    uniq, counts = uniq[order], counts[order]
    print(f"== {title} (k={len(uniq)}) ==")
    if len(uniq) <= max_groups:
        for u, c in zip(uniq, counts):
            print(f"- {u}: {c}")
    else:
        # In top-10 + tổng số nhóm
        for u, c in zip(uniq[:10], counts[:10]):
            print(f"- {u}: {c}")
        print(f"... ({len(uniq)-10} nhóm khác)")

# =========================
# Tiền xử lý + đặc trưng (NumPy-only)
# =========================
def _is_nan(x: Any) -> bool:
    try:
        return np.isnan(x)
    except Exception:
        return False

<<<<<<< HEAD
def _safe_to_float(a: np.ndarray) -> np.ndarray:
    out = a.astype(object)
    v = np.vectorize(lambda x: np.nan if (x is None or x == "" or (isinstance(x, str) and x.strip()=="" )) else x)
    out = v(out)
    return out.astype(float)

def _one_hot(cat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(cat.astype(object))
    idx = {u: i for i, u in enumerate(uniq)}
    O = np.zeros((cat.shape[0], len(uniq)), dtype=float)
    for r, v in enumerate(cat):
        O[r, idx[v]] = 1.0
    return O, uniq

def preprocess_and_save(
    csv_path: Path,
    out_npz: Path = PROCESSED_DIR / "airbnb_processed.npz",
    price_col: str = "price",
    borough_col: str = "neighbourhood_group",
    numeric_cols: Tuple[str, ...] = ("minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365"),
    categorical_cols: Tuple[str, ...] = ("neighbourhood_group", "room_type"),
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
) -> Path:
    """
    Pipeline NumPy-only:
    - Đọc CSV
    - Làm sạch: fill NA cho numeric, reviews_per_month -> 0 nếu NA
    - Clip ngoại lai theo percentiles
    - y = log1p(price)
    - One-hot cho cột phân loại
    - Chuẩn hoá X bằng z-score (fit theo toàn bộ tập vào cho HW2; nếu cần train/test split thì fit theo train)
    - Lưu .npz: X, y, feature_names, stats_mean, stats_std
    """
    header, data = load_csv_genfromtxt(csv_path)

    # map tên cột -> index
    col2idx = {name: i for i, name in enumerate(header.tolist())}

    # --- y = price -> log1p
    price_raw = _safe_to_float(data[:, col2idx[price_col]])
    y = np.log1p(price_raw)

    # --- numeric features
    feats = []
    feat_names = []

    for name in numeric_cols:
        col = _safe_to_float(data[:, col2idx[name]])
        # fill missing
        if name == "reviews_per_month":
            col = np.where(np.isnan(col), 0.0, col)
        else:
            # median fill
            med = np.nanmedian(col)
            col = np.where(np.isnan(col), med, col)

        # clip ngoại lai
        lo, hi = np.nanpercentile(col, clip_percentiles)
        col = np.clip(col, lo, hi)

        feats.append(col.reshape(-1, 1))
        feat_names.append(name)

    # --- categorical -> one-hot
    for name in categorical_cols:
        cat = data[:, col2idx[name]].astype(object)
        O, uniq = _one_hot(cat)
        feats.append(O)
        feat_names.extend([f"{name}=={u}" for u in uniq])

    X = np.hstack(feats)

    # --- standardize (z-score)
    mean = X.mean(axis=0, dtype=float)
    std = X.std(axis=0, dtype=float)
    std = np.where(std == 0, 1.0, std)
    X_std = (X - mean) / std

    # Lưu
    np.savez_compressed(
        out_npz,
        X=X_std,
        y=y.astype(float),
        feature_names=np.array(feat_names, dtype=object),
        stats_mean=mean,
        stats_std=std
    )
    _echo(f"Đã lưu: {out_npz} (X:{X_std.shape}, y:{y.shape})")
    return out_npz

# =========================
# Check nhanh dataset
# =========================
def load_and_check(root: str | Path = RAW_DIR, filename: str | None = None) -> Dict[str, Any]:
    """
    Kiểm tra nhanh các nhóm (ví dụ borough) để in đủ tên nhóm khi nhóm ít.
    """
    root = Path(root)
    if filename is None:
        # đoán file CSV đầu tiên trong raw
        csvs = sorted(root.glob("*.csv")) + sorted(root.glob("*.csv.gz"))
        if not csvs:
            raise FileNotFoundError(f"Không tìm thấy CSV trong {root}")
        csv_path = csvs[0]
    else:
        csv_path = root / filename

    header, data = load_csv_genfromtxt(csv_path)
    col2idx = {name: i for i, name in enumerate(header.tolist())}

    if "neighbourhood_group" in col2idx:
        borough = data[:, col2idx["neighbourhood_group"]].astype(object)
        print_small_category_full(borough, max_groups=10, title="Boroughs")

    info = {
        "rows": data.shape[0],
        "cols": data.shape[1],
        "columns": header.tolist(),
        "csv_path": str(csv_path)
    }
    return info
=======
    # availability_365 = số ngày sẵn sàng trong 365 ngày tới
    availability = num_data.get("availability_365", np.array([], dtype=np.int64))
    out_of_range = int(np.sum((availability < 0) | (availability > 365))) if availability.size else 0

    rpm = num_data.get("reviews_per_month", np.array([], dtype=np.float64))
    na_rpm = int(np.sum(np.isnan(rpm))) if rpm.size else 0

    last_review = text_data.get("last_review", np.array([], dtype=object))
    if last_review.size:
        mask = np.array([str(val).strip() in {"", "NA"} for val in last_review], dtype=bool)
        na_last = int(np.sum(mask))
    else:
        na_last = 0

    return {
        "n_rows": total_rows,
        "n_cols": len(header),
        "columns": header,
        "uniq_neigh_group": _unique_with_sample(neigh),
        "uniq_room_type": _unique_with_sample(room),
        "out_of_range_avail": out_of_range,
        "na_reviews_per_month": na_rpm,
        "na_last_review": na_last,
    }


def load_and_check(root: str = "data") -> Dict[str, object]:
    dirs = ensure_data_dirs(root=root)
    csv_path = dirs["raw"] / FILENAME
    
    if not csv_path.exists():
        kaggle_download_if_needed(
            dataset=DATASET_NAME,
            filename=FILENAME,
            out_dir=str(dirs["raw"]),
        )

    data = load_airbnb_numpy(csv_path)
    report = basic_checks(data)

    print(f"Shape: {report['n_rows']} hang x {report['n_cols']} cot")
    
    key_cols = [col for col in KEY_COLUMNS if col in report["columns"]]
    print(f"Important Columns: {', '.join(key_cols)}")
    
    neigh = report["uniq_neigh_group"]
    print(f"neighbourhood_group: {neigh['count']} groups -> {', '.join(map(str, neigh['values']))}")
    
    room = report["uniq_room_type"]
    print(f"room_type: {room['count']} types -> {', '.join(map(str, room['values']))}")
    print(f"availability_365 out of [0, 365]: {report['out_of_range_avail']}")
    print(f"reviews_per_month NA: {report['na_reviews_per_month']}")
    print(f"last_review NA: {report['na_last_review']}")

    return report


if __name__ == "__main__":
    load_and_check()
>>>>>>> parent of d1a2c48 (move to ./src)
