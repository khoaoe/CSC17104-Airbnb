import csv
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

DATASET_NAME = "dgomonov/new-york-city-airbnb-open-data"
FILENAME = "AB_NYC_2019.csv"

KEY_COLUMNS = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "latitude",
    "longitude",
]


def ensure_data_dirs(root: str = "data") -> Dict[str, Path]:
    base = Path(root)
    
    raw = base / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    
    processed = base / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    
    return {"root": base, "raw": raw, "processed": processed}


def kaggle_download_if_needed(dataset: str, filename: str, out_dir: str) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / filename
    if target.exists():
        return target

    cmd = [
        "../.venv/bin/kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(out_path),
        "--quiet",
        "--unzip",
    ]
    
    try:
        # Kaggle CLI cần ~/.kaggle/kaggle.json hoặc biến môi trường KAGGLE_USERNAME/KAGGLE_KEY
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Không tìm thấy Kaggle CLI") from exc

    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        if not msg:
            msg = "Không tải được dữ liệu từ Kaggle."
        if "401" in msg or "403" in msg or "Unauthorized" in msg:
            raise RuntimeError("Kaggle CLI chưa cấu hình. Kiểm tra kaggle.json hoặc biến môi trường") from None
        raise RuntimeError(f"Tải từ Kaggle thất bại: {msg}")

    zip_path = out_path / f"{filename}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_path)
        zip_path.unlink()

    if not target.exists():
        raise RuntimeError(f"Tải Kaggle xong nhưng thiếu file {filename}")
    return target


def _convert_float(values: Iterable[str]) -> np.ndarray:
    data = list(values)
    arr = np.empty(len(data), dtype=np.float64)
    for idx, raw_val in enumerate(data):
        val = raw_val.strip()
        if val == "" or val.upper() == "NA":
            arr[idx] = np.nan
        else:
            arr[idx] = float(val)
    return arr


def _convert_int(values: Iterable[str]) -> np.ndarray:
    data = list(values)
    arr = np.empty(len(data), dtype=np.int64)
    for idx, raw_val in enumerate(data):
        val = raw_val.strip()
        if val == "" or val.upper() == "NA":
            raise ValueError("Giá trị NA xuất hiện ở cột số nguyên.")
        arr[idx] = int(float(val))
    return arr


def load_airbnb_numpy(csv_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError("File CSV trong") from exc
        rows = [row for row in reader]

    columns: Dict[str, List[str]] = {col: [] for col in header}
    for row in rows:
        for col, value in zip(header, row):
            columns[col].append(value)

    floats = {
        "price",
        "reviews_per_month",
        "latitude",
        "longitude",
    }
    ints = {
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
    }

    num: Dict[str, np.ndarray] = {}
    text: Dict[str, np.ndarray] = {}

    for col, values in columns.items():
        if col in floats:
            num[col] = _convert_float(values)
        elif col in ints:
            num[col] = _convert_int(values)
        else:
            text[col] = np.array(values, dtype=object)

    return {"header": header, "text": text, "num": num}


def _unique_with_sample(arr: np.ndarray, max_samples: int = 4) -> Dict[str, object]:
    if arr.size == 0:
        return {"count": 0, "values": []}
    unique_vals = np.unique(arr)
    sample = unique_vals[:max_samples].tolist()
    return {"count": int(unique_vals.size), "values": sample}


def basic_checks(data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, object]:
    header = data["header"]
    num_data = data["num"]
    text_data = data["text"]

    if num_data:
        total_rows = len(next(iter(num_data.values())))
    elif text_data:
        total_rows = len(next(iter(text_data.values())))
    else:
        total_rows = 0

    neigh = text_data.get("neighbourhood_group", np.array([], dtype=object))
    room = text_data.get("room_type", np.array([], dtype=object))

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


def clip_outliers_percentile(
    x: np.ndarray, p_low: float = 0.0, p_high: float = 99.5
) -> Tuple[np.ndarray, float, float]:
    """Clip theo phan tram vi de cat duoi cuc tri."""
    lo, hi = np.percentile(x, [p_low, p_high])
    return np.clip(x, lo, hi), float(lo), float(hi)


def one_hot(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One-hot cho mang object/string, tra ve (matrix, uniques)."""
    uniq, inv = np.unique(values, return_inverse=True)
    oh = np.eye(uniq.size, dtype=np.float64)[inv]
    return oh, uniq


def nonempty_mask(arr_obj: np.ndarray) -> np.ndarray:
    """True neu text khong rong hoac la 'NA'."""
    return np.array([str(v).strip() not in {"", "NA"} for v in arr_obj], dtype=bool)


def safe_log1p(x: np.ndarray) -> np.ndarray:
    """log1p an toan cho gia tri khong am."""
    x = np.maximum(x, 0.0)
    return np.log1p(x)


@dataclass
class PreprocessConfig:
    p_high_min_nights: float = 99.0
    p_high_host_listings: float = 99.0
    host_big_threshold: float = 3.0


@dataclass
class PreprocessBundle:
    X: np.ndarray
    y: np.ndarray
    y_log1p: np.ndarray
    feature_names: np.ndarray
    room_type_uniques: np.ndarray
    neigh_group_uniques: np.ndarray


def build_features(
    num: Dict[str, np.ndarray],
    txt: Dict[str, np.ndarray],
    cfg: PreprocessConfig = PreprocessConfig(),
) -> PreprocessBundle:
    """Build tap dac trung NumPy-only tu dict cot so/text."""
    price = num["price"].astype(np.float64)
    minimum_nights = num["minimum_nights"].astype(np.float64)
    number_of_reviews = num["number_of_reviews"].astype(np.float64)
    host_listings = num["calculated_host_listings_count"].astype(np.float64)
    availability_365 = np.clip(num["availability_365"].astype(np.float64), 0, 365)
    reviews_per_month = num["reviews_per_month"].astype(np.float64)

    room_type = txt["room_type"]
    neigh_group = txt["neighbourhood_group"]
    last_review = txt["last_review"]

    rpm_filled = reviews_per_month.copy()
    rpm_filled[np.isnan(rpm_filled)] = 0.0

    min_nights_clip, _, _ = clip_outliers_percentile(
        minimum_nights, 0.0, cfg.p_high_min_nights
    )
    host_listings_clip, _, _ = clip_outliers_percentile(
        host_listings, 0.0, cfg.p_high_host_listings
    )

    has_last_review = nonempty_mask(last_review).astype(np.float64)
    is_entire_home = (room_type == "Entire home/apt").astype(np.float64)
    host_is_big = (host_listings_clip >= cfg.host_big_threshold).astype(np.float64)

    oh_room, rt_uniques = one_hot(room_type)
    oh_neigh, ng_uniques = one_hot(neigh_group)

    X_blocks = [
        safe_log1p(number_of_reviews).reshape(-1, 1),
        safe_log1p(min_nights_clip).reshape(-1, 1),
        safe_log1p(host_listings_clip).reshape(-1, 1),
        (availability_365 / 365.0).reshape(-1, 1),
        safe_log1p(rpm_filled).reshape(-1, 1),
        is_entire_home.reshape(-1, 1),
        host_is_big.reshape(-1, 1),
        has_last_review.reshape(-1, 1),
        oh_room,
        oh_neigh,
    ]
    X = np.concatenate(X_blocks, axis=1)
    y = price.copy()
    y_log1p = safe_log1p(y)

    feature_names = [
        "log1p_num_reviews",
        "log1p_minimum_nights",
        "log1p_host_listings",
        "days_available_ratio",
        "log1p_reviews_per_month",
        "is_entire_home",
        "host_is_big_ge3",
        "has_last_review",
    ] + [f"rt::{s}" for s in rt_uniques.tolist()] + [
        f"ng::{s}" for s in ng_uniques.tolist()
    ]

    return PreprocessBundle(
        X=X,
        y=y,
        y_log1p=y_log1p,
        feature_names=np.array(feature_names, dtype=object),
        room_type_uniques=rt_uniques.astype(object),
        neigh_group_uniques=ng_uniques.astype(object),
    )


def preprocess_and_save(
    root: str = "data",
    out_name: str = "airbnb_2019_preprocessed.npz",
    cfg: PreprocessConfig = PreprocessConfig(),
) -> Path:
    """End-to-end: load CSV tho, build features, luu npz."""
    dirs = ensure_data_dirs(root=root)
    csv_path = dirs["raw"] / FILENAME
    if not csv_path.exists():
        kaggle_download_if_needed(
            dataset=DATASET_NAME, filename=FILENAME, out_dir=str(dirs["raw"])
        )
    data = load_airbnb_numpy(csv_path)
    bundle = build_features(data["num"], data["text"], cfg)

    out_path = dirs["processed"] / out_name
    np.savez_compressed(
        out_path,
        X=bundle.X,
        y=bundle.y,
        y_log1p=bundle.y_log1p,
        feature_names=bundle.feature_names,
        room_type_uniques=bundle.room_type_uniques,
        neigh_group_uniques=bundle.neigh_group_uniques,
    )
    return out_path


if __name__ == "__main__":
    load_and_check()
