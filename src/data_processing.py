# Chỉ dùng NumPy cho xử lý dữ liệu
from __future__ import annotations
import os
import glob
from typing import Dict, Iterable, Tuple, List, Optional
import numpy as np
from datetime import datetime
import csv

# --- Cấu hình cột mặc định cho dataset AB_NYC_2019 ---
numeric_cols = [
    "id", "host_id", "latitude", "longitude", "price",
    "minimum_nights", "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
]
categorical_cols = [
    "name", "host_name", "neighbourhood_group",
    "neighbourhood", "room_type",
]
date_cols = {"last_review"}


# --- Tìm file CSV trong thư mục (ưu tiên AB_NYC_2019.csv) ---
def find_csv(root: str) -> str:
    """Tìm file .csv trong thư mục root; ưu tiên 'AB_NYC_2019.csv'"""
    cand = os.path.join(root, "AB_NYC_2019.csv")
    if os.path.isfile(cand):
        return cand
    files = sorted(glob.glob(os.path.join(root, "*.csv")))
    if not files:
        raise FileNotFoundError(f"Không thấy file .csv trong: {root}")
    return files[0]


# --- Parse cột ngày sang datetime64[D] ---
def _parse_date_col(s: np.ndarray) -> np.ndarray:
    """
    Chuyển mảng string 'YYYY-MM-DD' -> datetime64[D]; rỗng -> NaT
    """
    out = np.empty(s.shape, dtype="datetime64[D]")
    # mask rỗng
    m_empty = (s == "") | (s == "nan") | (s == "NaN")
    out[m_empty] = np.datetime64("NaT")
    if (~m_empty).any():
        out[~m_empty] = s[~m_empty].astype("datetime64[D]")
    return out


# --- Chuyển structured array -> dict cột (1D ndarray mỗi cột) ---
def to_columns(arr: np.ndarray) -> Dict[str, np.ndarray]:
    cols = {name: arr[name] for name in arr.dtype.names}
    # Chuẩn hoá cột ngày nếu có
    for dcol in date_cols:
        if dcol in cols and cols[dcol].dtype.kind in ("U", "O"):
            cols[dcol] = _parse_date_col(cols[dcol].astype(str))
    return cols


# --- Đọc CSV thành dict cột với dtype phù hợp ---
def _safe_float(x: str) -> float:
    # rỗng -> NaN
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        # Trường hợp bẩn hiếm gặp -> NaN để không vỡ quy trình
        return float("nan")

def _safe_date(x: str):
    if not x:
        return np.datetime64("NaT")
    try:
        # dữ liệu dùng 'YYYY-MM-DD'
        return np.datetime64(datetime.strptime(x, "%Y-%m-%d").date())
    except Exception:
        return np.datetime64("NaT")
    
def _read_csv_dict(path: str) -> Dict[str, np.ndarray]:
    """
    Đọc CSV bằng csv.DictReader để xử lý đúng dấu phẩy trong chuỗi được quote
    Trả về dict {column -> ndarray} với dtype phù hợp
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Không đọc được header CSV")
        # Chuẩn hoá tên cột (giữ nguyên như trong file)
        cols_py = {k: [] for k in fieldnames}

        for row in reader:
            for k in fieldnames:
                v = row.get(k, "")
                if k in numeric_cols:
                    cols_py[k].append(_safe_float(v))
                elif k in date_cols:
                    cols_py[k].append(_safe_date(v))
                else:
                    # giữ string; strip khoảng trắng dư cho gọn
                    cols_py[k].append(v.strip() if isinstance(v, str) else v)

    # Chuyển list -> numpy array với dtype hợp lý
    out: Dict[str, np.ndarray] = {}
    for k, lst in cols_py.items():
        if k in numeric_cols:
            out[k] = np.array(lst, dtype=float)
        elif k in date_cols:
            out[k] = np.array(lst, dtype="datetime64[D]")
        else:
            out[k] = np.array(lst, dtype="U")  # unicode string
    return out


# --- API load chính cho dataset ---
def load_airbnb(path_or_dir: str) -> Dict[str, np.ndarray]:
    """
    Load dataset Airbnb NYC 2019 (CSV chuẩn, có quoted fields)
    - Nếu truyền thư mục: tự tìm file CSV
    - Trả về: dict {column -> ndarray}
    """
    csv_path = path_or_dir
    if os.path.isdir(path_or_dir):
        csv_path = find_csv(path_or_dir)
    return _read_csv_dict(csv_path)


# --- Thống kê missing theo cột ---
def missing_summary(cols: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Trả về bảng [column, missing_count, missing_rate_%] (dtype object/float)
    """
    names, miss_cnt, miss_rate = [], [], []
    n = len(next(iter(cols.values())))
    for k, v in cols.items():
        if v.dtype.kind == "M":  # datetime64
            m = np.isnat(v)
        else:
            if v.dtype.kind in ("U", "S", "O"):
                m = (v == "") | (v.astype(str) == "nan")
            else:
                m = np.isnan(v) if np.issubdtype(v.dtype, np.floating) else np.zeros(n, bool)
        names.append(k)
        mc = int(m.sum())
        miss_cnt.append(mc)
        miss_rate.append(100.0 * mc / n)
    out = np.empty(len(names), dtype=[("column", "U50"), ("missing_count", "i8"), ("missing_rate_%", "f8")])
    out["column"] = names
    out["missing_count"] = miss_cnt
    out["missing_rate_%"] = miss_rate
    return out


# --- Đếm unique cho cột phân loại ---
def unique_summary(cols: Dict[str, np.ndarray], cat_cols: Iterable[str]) -> np.ndarray:
    """
    Bảng [column, unique_count, unique_rate_%] cho các cột phân loại
    """
    names, ucnt, urate = [], [], []
    n = len(next(iter(cols.values())))
    for k in cat_cols:
        if k not in cols:
            continue
        u = np.unique(cols[k].astype(str))
        names.append(k)
        ucnt.append(len(u))
        urate.append(100.0 * len(u) / n)
    out = np.empty(len(names), dtype=[("column", "U50"), ("unique_count", "i8"), ("unique_rate_%", "f8")])
    out["column"] = names
    out["unique_count"] = ucnt
    out["unique_rate_%"] = urate
    return out


# --- Describe nhanh cho cột số ---
def describe_numeric(cols: Dict[str, np.ndarray], num_cols: Iterable[str]) -> np.ndarray:
    """
    Trả về bảng thống kê cơ bản cho cột số: min, p25, p50, p75, max, mean, std
    """
    stats = []
    dtype = [
        ("column", "U40"), ("min", "f8"), ("p25", "f8"), ("p50", "f8"),
        ("p75", "f8"), ("max", "f8"), ("mean", "f8"), ("std", "f8"),
    ]
    for k in num_cols:
        if k not in cols:
            continue
        x = cols[k].astype(float)
        # bỏ NaN nếu có
        xm = x[~np.isnan(x)]
        if xm.size == 0:
            stats.append((k, *([np.nan] * 7)))
            continue
        q = np.percentile(xm, [0, 25, 50, 75, 100])
        stats.append((k, q[0], q[1], q[2], q[3], q[4], float(xm.mean()), float(xm.std(ddof=1))))
    out = np.array(stats, dtype=dtype)
    return out


# --- Groupby + reduce nhanh (mean/sum/count) ---
def groupby_reduce(keys: np.ndarray, values: np.ndarray, how: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    """
    Nhóm theo 'keys' (string hoặc số) và gom 'values' bằng mean/sum/count
    """
    uk, inv = np.unique(keys, return_inverse=True)
    if how == "count":
        agg = np.bincount(inv).astype(float)
        return uk, agg
    # chuẩn hoá values -> float + bỏ NaN
    v = values.astype(float)
    v[np.isnan(v)] = 0.0
    sums = np.bincount(inv, weights=v)
    if how == "sum":
        return uk, sums
    counts = np.bincount(inv, weights=(~np.isnan(values.astype(float))).astype(float))
    # tránh chia 0
    counts[counts == 0] = np.nan
    means = sums / counts
    return uk, means


# --- Top-k theo tần suất ---
def topk_counts(a: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trả về (giá trị, count) theo tần suất giảm dần, lấy top-k
    """
    vals, counts = np.unique(a.astype(str), return_counts=True)
    order = np.argsort(-counts)
    vals, counts = vals[order], counts[order]
    k = min(k, vals.size)
    return vals[:k], counts[:k]


# --- Ma trận tương quan cho tập cột số ---
def corr_matrix(cols: Dict[str, np.ndarray], num_cols: Iterable[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Tính corrcoef (Pearson) giữa các cột số; tự động bỏ NaN theo hàng
    """
    keep = [c for c in num_cols if c in cols]
    X = np.column_stack([cols[c].astype(float) for c in keep])
    # mask hàng có NaN
    m = ~np.isnan(X).any(axis=1)
    X = X[m]
    if X.shape[0] < 2:
        return np.full((len(keep), len(keep)), np.nan), keep
    C = np.corrcoef(X, rowvar=False)
    return C, keep


# --- Lọc theo khung toạ độ NYC để loại record sai/ngoài vùng ---
def filter_geo_bounds(
    cols: Dict[str, np.ndarray],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    lat_range: Tuple[float, float] = (40.5, 40.9),
    lon_range: Tuple[float, float] = (-74.25, -73.7),
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Giữ các dòng có (lat, lon) nằm trong khung NYC thường dùng
    Trả về (cols_filtered, mask_kept)
    """
    lat = cols[lat_col].astype(float)
    lon = cols[lon_col].astype(float)
    m = (lat >= lat_range[0]) & (lat <= lat_range[1]) & (lon >= lon_range[0]) & (lon <= lon_range[1])
    kept = {k: v[m] for k, v in cols.items()}
    return kept, m


# --- Điền thiếu cho reviews_per_month dựa trên logic "chưa có review" ---
def fill_reviews_per_month_zero(cols: Dict[str, np.ndarray]) -> None:
    """
    Nếu number_of_reviews == 0 -> reviews_per_month = 0 (nếu đang NaN)
    Giữ nguyên các trường hợp còn lại (để tránh bias)
    """
    if "reviews_per_month" not in cols or "number_of_reviews" not in cols:
        return
    rpm = cols["reviews_per_month"].astype(float)
    nor = cols["number_of_reviews"].astype(float)
    m = np.isnan(rpm) & (nor == 0)
    if m.any():
        rpm[m] = 0.0
        cols["reviews_per_month"] = rpm  # in-place update


# --- Điền thiếu bằng median cho các cột số được chỉ định ---
def impute_numeric_median(cols: Dict[str, np.ndarray], numeric_cols: Iterable[str]) -> Dict[str, float]:
    """
    Impute median cho NaN ở các cột số; trả về dict {col: median_used} để log
    """
    used = {}
    for c in numeric_cols:
        if c not in cols:
            continue
        x = cols[c].astype(float)
        if np.isnan(x).any():
            med = float(np.nanmedian(x))
            x[np.isnan(x)] = med
            cols[c] = x
            used[c] = med
    return used


# --- Cắt ngoại lai theo percentile (winsorize nhẹ) ---
def clip_outliers_percentile(x: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
    """
    Trả về bản sao đã kẹp giá trị về [q_low, q_high] theo percentile
    Dùng cho các biến lệch/phân phối nặng đuôi như price, minimum_nights
    """
    x = x.astype(float).copy()
    ql, qh = np.nanpercentile(x, [low_q, high_q])
    x[x < ql] = ql
    x[x > qh] = qh
    return x


# --- Mã hoá phân loại -> id số (0..K-1), gom nhãn hiếm vào '__OTHER__' ---
def fit_category_encoder(values: np.ndarray, min_count: int = 1) -> Tuple[Dict[str, int], int]:
    """
    Tạo mapping {category -> id}. Nhãn xuất hiện < min_count -> gom vào '__OTHER__'
    Trả về (mapping, n_classes). '__OTHER__' chỉ có nếu có nhãn hiếm
    """
    vals = values.astype(str)
    uniq, counts = np.unique(vals, return_counts=True)
    mapping: Dict[str, int] = {}
    next_id = 0
    has_other = False
    for v, c in zip(uniq, counts):
        if c >= min_count:
            mapping[v] = next_id
            next_id += 1
        else:
            has_other = True
    if has_other:
        mapping["__OTHER__"] = next_id
        next_id += 1
    return mapping, next_id


def transform_category(values: np.ndarray, mapping: Dict[str, int]) -> np.ndarray:
    """
    Biến mảng string -> mảng id số theo mapping. Nhãn lạ -> '__OTHER__' nếu có, ngược lại -> -1
    """
    vals = values.astype(str)
    has_other = "__OTHER__" in mapping
    other_id = mapping.get("__OTHER__", -1)
    out = np.empty(vals.shape[0], dtype=int)
    for i, v in enumerate(vals):
        out[i] = mapping.get(v, other_id if has_other else -1)
    return out


# --- One-hot từ mã số ---
def one_hot(codes: np.ndarray, n_classes: int) -> np.ndarray:
    """
    One-hot encode: shape (n_samples, n_classes). codes ngoài [0..K-1] -> hàng zero
    """
    n = codes.shape[0]
    O = np.zeros((n, n_classes), dtype=float)
    m = (codes >= 0) & (codes < n_classes)
    O[np.arange(n)[m], codes[m]] = 1.0
    return O


# --- Lắp ma trận đặc trưng X và nhãn y (ví dụ: y=price) ---
def assemble_features_airbnb(
    cols: Dict[str, np.ndarray],
    *,
    num_cols: Iterable[str] = (
        "latitude", "longitude",
        "minimum_nights", "number_of_reviews", "reviews_per_month",
        "calculated_host_listings_count", "availability_365",
    ),
    cat_cols: Iterable[str] = ("neighbourhood_group", "room_type"),
    cat_min_count: int = 10,   # gom nhãn rất hiếm
    target_col: str = "price",
    clip_target_percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Dict[str, int]]]:
    """
    Trả về (X, y, feature_names, encoders).
    - num_cols: dùng trực tiếp (sau khi impute NaN bên ngoài)
    - cat_cols: mã hoá one-hot (gom nhãn hiếm về '__OTHER__')
    - target_col: y; có thể clip nhẹ để giảm ảnh hưởng ngoại lai
    """
    feats: List[np.ndarray] = []
    names: List[str] = []

    # numeric
    for c in num_cols:
        if c in cols:
            x = cols[c].astype(float)
            feats.append(x.reshape(-1, 1))
            names.append(c)

    # categorical -> one-hot
    encoders: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        if c in cols:
            mapping, K = fit_category_encoder(cols[c], min_count=cat_min_count)
            encoders[c] = mapping
            codes = transform_category(cols[c], mapping)
            O = one_hot(codes, K)
            feats.append(O)
            # thêm tên cột theo thứ tự id
            inv = sorted([(v, k) for k, v in mapping.items()], key=lambda t: t[0])
            names.extend([f"{c}={lab}" for _, lab in inv])

    X = np.column_stack(feats) if feats else np.empty((len(next(iter(cols.values()))), 0), float)

    # target
    y = cols[target_col].astype(float).copy()
    if clip_target_percentiles is not None:
        y = clip_outliers_percentile(y, *clip_target_percentiles)

    return X, y, names, encoders

# --- Geo features ---
def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Khởi tạo kmeans++ đơn giản cho ổn định"""
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=float)
    i0 = rng.integers(0, n)
    centroids[0] = X[i0]
    d2 = np.sum((X - centroids[0])**2, axis=1)

    for i in range(1, k):
        probs = d2 / np.sum(d2)
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centroids[i])**2, axis=1))
    return centroids

def kmeans_fit_np(
    X: np.ndarray, k: int, *, max_iter: int = 50, tol: float = 1e-4, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KMeans tối giản cho 2D (cũng chạy được >2D):
    Trả về (labels shape (n,), centroids shape (k, d)).
    """
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    C = _kmeans_pp_init(X, k, rng)

    last_inertia = None
    labels = np.zeros(X.shape[0], dtype=int)
    for _ in range(max_iter):
        # assign
        # (n,k) khoảng cách bình phương đến từng centroid
        d2 = np.sum((X[:, None, :] - C[None, :, :])**2, axis=2)
        labels = np.argmin(d2, axis=1)
        # update
        C_new = C.copy()
        for j in range(k):
            idx = (labels == j)
            if idx.any():
                C_new[j] = X[idx].mean(axis=0)
        # hội tụ
        inertia = np.sum((X - C_new[labels])**2)
        if (last_inertia is not None) and (abs(last_inertia - inertia) <= tol * max(1.0, last_inertia)):
            C = C_new
            break
        C = C_new
        last_inertia = inertia
    return labels, C

def geo_kmeans_features_from_cols(
    cols: Dict[str, np.ndarray], *, k: int = 8, lat_col: str = "latitude", lon_col: str = "longitude", seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Tạo đặc trưng geo từ (lat, lon):
      - 'geo_cluster' (id 0..k-1)
      - 'geo_d2c_j' = khoảng cách Euclid tới centroid j (j=0..k-1)
    """
    lat = cols[lat_col].astype(float)
    lon = cols[lon_col].astype(float)
    Xgeo = np.column_stack([lat, lon])
    labels, C = kmeans_fit_np(Xgeo, k=k, seed=seed)

    # khoảng cách tới từng centroid
    d2 = np.sum((Xgeo[:, None, :] - C[None, :, :])**2, axis=2)**0.5  # (n,k)
    out = {"geo_cluster": labels.astype(int)}
    for j in range(C.shape[0]):
        out[f"geo_d2c_{j}"] = d2[:, j].astype(float)
    return out

# --- Features X và label y + geo features ---
from typing import Dict, Tuple, List
import numpy as np

def assemble_features_airbnb_plus_geo(
    cols: Dict[str, np.ndarray],
    *,
    num_cols: Tuple[str, ...] = (
        "latitude", "longitude", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count", "availability_365",
    ),
    cat_cols: Tuple[str, ...] = (
        "neighbourhood_group", "room_type", "neighbourhood",
    ),
    cat_min_count: int = 10,
    target_col: str = "price",
    clip_target_percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    k_geo: int = 8,  # số cluster geo
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Dict[str, int]]]:
    """
    Giống assemble_features_airbnb nhưng bổ sung:
      - numeric: 'dist_center' (khoảng cách đến trung tâm thành phố)
      - categorical: 'ngrt' = neighbourhood_group __ room_type
      - geo features: one-hot cluster + khoảng cách tới centroid
    """
    # Copy cols để không phá dict gốc bên ngoài
    cols_ext: Dict[str, np.ndarray] = dict(cols)

    lat = cols_ext["latitude"].astype(float)
    lon = cols_ext["longitude"].astype(float)
    ng  = cols_ext["neighbourhood_group"].astype(str)
    rt  = cols_ext["room_type"].astype(str)

    # 1) Distance to center (ví dụ Times Square)
    center_lat, center_lon = 40.7580, -73.9855  # Times Square
    dist_center = np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)
    cols_ext["dist_center"] = dist_center

    # 2) Interaction borough × room_type
    ngrt = np.char.add(ng, "__")
    ngrt = np.char.add(ngrt, rt)
    cols_ext["ngrt"] = ngrt

    # Chuyển num_cols / cat_cols thành list để append được
    num_cols_list = list(num_cols) + ["dist_center"]
    cat_cols_list = list(cat_cols) + ["ngrt"]

    # 3) Base features từ assemble_features_airbnb
    X_base, y, names, encoders = assemble_features_airbnb(
        cols_ext,
        num_cols=tuple(num_cols_list),
        cat_cols=tuple(cat_cols_list),
        cat_min_count=cat_min_count,
        target_col=target_col,
        clip_target_percentiles=clip_target_percentiles,
    )

    # 4) Geo features từ lat/lon
    geo = geo_kmeans_features_from_cols(cols_ext, k=k_geo)
    # one-hot cluster
    mapping_geo, K = fit_category_encoder(geo["geo_cluster"], min_count=1)
    codes_geo = transform_category(geo["geo_cluster"], mapping_geo)
    O_geo = one_hot(codes_geo, K)

    # distances đến các centroid (các cột geo_d2c_*)
    d_cols = [v for k, v in geo.items() if k.startswith("geo_d2c_")]
    D = np.column_stack(d_cols) if d_cols else np.empty((X_base.shape[0], 0), float)

    # 5) Gộp tất cả vào X
    X = np.column_stack([X_base, O_geo, D])

    # Tên cột cho geo_cluster one-hot
    inv = sorted([(v, k) for k, v in mapping_geo.items()], key=lambda t: t[0])
    names_geo = [f"geo_cluster={lab}" for _, lab in inv]

    # Tên cột cho các khoảng cách đến centroid (theo thứ tự d_cols)
    dist_names = [f"geo_d2c_{i}" for i in range(len(d_cols))]

    names = names + names_geo + dist_names

    encoders["geo_cluster"] = mapping_geo
    return X, y, names, encoders


# --- Gói gọn pipeline tiền xử lý thường dùng ---
def preprocess_airbnb_default(
    cols: Dict[str, np.ndarray],
    *,
    geo_filter: bool = True,
    impute_median_for: Iterable[str] = (
        "minimum_nights", "number_of_reviews", "reviews_per_month",
        "calculated_host_listings_count", "availability_365",
    ),
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """
    Pipeline ngắn gọn:
      1) (tuỳ chọn) lọc geo theo khung NYC
      2) điền reviews_per_month = 0 nếu chưa có review
      3) impute median cho các cột số còn thiếu
    Trả về (cols_clean, report_dict).
    """
    report = {}

    # 1) geo filter
    if geo_filter:
        cols2, m = filter_geo_bounds(cols)
        report["kept_geo_ratio"] = float(m.mean())
        cols = cols2

    # 2) logic review
    fill_reviews_per_month_zero(cols)

    # 3) impute median
    used = impute_numeric_median(cols, impute_median_for)
    report["median_imputed"] = used
    return cols, report


# --- Lưu gói dữ liệu đã xử lý ra .npz ---
def save_processed_npz(
    path: str,
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    encoders: Optional[Dict[str, Dict[str, int]]] = None,
    meta: Optional[Dict[str, object]] = None,
) -> None:
    """
    Lưu các thành phần cần cho modeling sau: X, y, tên cột, encoder, meta.
    """
    np.savez_compressed(
        path,
        X=X,
        y=y,
        feature_names=np.array(feature_names, dtype="U"),
        encoders=np.array(encoders if encoders is not None else {}, dtype=object),
        meta=np.array(meta if meta is not None else {}, dtype=object),
    )

# --- Đọc gói dữ liệu đã xử lý từ .npz ---
def load_processed_npz(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], dict, dict]:
    """
    Đọc file .npz đã lưu từ bước tiền xử lý.
    Trả về: (X, y, feature_names, encoders, meta)
    """
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y = d["y"]
    feature_names = d["feature_names"].astype(str).tolist()
    encoders = dict(d["encoders"].item()) if "encoders" in d.files else {}
    meta = dict(d["meta"].item()) if "meta" in d.files else {}
    return X, y, feature_names, encoders, meta