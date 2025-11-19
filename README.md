# [CSC17104 - Lab02] Airbnb NYC 2019

Dự án tuân theo yêu cầu  **NumPy-only** cho tiền xử lý và xây dựng mô hình dự đoán cho dataset **New York City Airbnb Open Data (2019)**

---

## Mục lục

* [Giới thiệu](#giới-thiệu)
* [Dataset](#dataset)
* [Method](#method)
* [Installation & Setup](#installation--setup)
* [Usage](#usage)
* [Results](#results)
* [Project Structure](#project-structure)
* [Challenges & Solutions](#challenges--solutions)
* [Future Improvements](#future-improvements)
* [Contributors](#contributors)
* [License](#license)

---

## Giới thiệu

**Bài toán** Dự đoán **giá thuê/đêm** (USD) của listing Airbnb tại New York dựa trên thông tin của listing (vị trí, loại phòng, lượt review, ...)

**Động lực & ứng dụng**

- Giúp chủ sở hữu định giá hợp lý
- Nền tảng gợi ý giá
- Phục vụ nghiên cứu, phân tích yếu tố ảnh hưởng đến giá cho thuê

**Mục tiêu cụ thể**

1. Xây dựng pipeline **EDA → Preprocessing → Modeling** chỉ dùng **NumPy**
2. Tạo đặc trưng vị trí, giảm lệch đuôi bằng log-transform.
3. Huấn luyện mô hình hồi quy tuyến tính (Ridge) + chọn siêu tham số bằng K-Fold CV
4. Báo cáo metric (RMSE/MAE/R²) và trực quan hoá kết quả

---

## Dataset

* **Nguồn:** [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) — `dgomonov/new-york-city-airbnb-open-data` (file data cần thiết: `AB_NYC_2019.csv`)
* **Kích thước:** ~48.9k dòng, 16 cột.
* **Các feature chính (trích):**

  * Định danh: `id`, `host_id`, `host_name`, `name`
  * Vị trí: `neighbourhood_group` (5 quận), `neighbourhood`, `latitude`, `longitude`
  * Thuộc tính: `room_type`, `minimum_nights`, `availability_365`
  * Đánh giá: `number_of_reviews`, `last_review`, `reviews_per_month`
  * **Target:** `price` (USD/đêm)

**Đặc điểm dữ liệu:**

* cần chuẩn hóa, **có quoting** (nhiều chuỗi chứa dấu phẩy trong `"..."`)
* Phân phối **lệch phải** (giá có ngoại lai cao)
* Thiếu dữ liệu ở `last_review`, `reviews_per_month` (đặc biệt khi `number_of_reviews = 0`)

---

## Method

### Quy trình xử lý dữ liệu (NumPy-only)

1. **Đọc CSV chuẩn RFC** bằng `csv.DictReader` → dict `{cột → ndarray}`
2. **Lọc geo** theo khung NYC (lat: [40.5, 40.9], lon: [-74.25, -73.7])
3. **Điền thiếu có ngữ nghĩa:** nếu `number_of_reviews==0` ⇒ `reviews_per_month=0`; còn lại impute **median** cho cột số
4. **Giảm ngoại lai:** clip nhẹ theo percentiles (mặc định [1, 99]) cho **target**
5. **Mã hoá phân loại:** one-hot cho `neighbourhood_group`, `room_type`, `neighbourhood` (gộp nhãn hiếm)
6. **Đặc trưng vị trí nâng cao:**
   * **KMeans (NumPy)** trên (lat, lon) → `geo_cluster` (one-hot) + **khoảng cách tới các centroid**
7. Gói `X, y, feature_names, encoders, meta` → `.npz`


### Thuật toán & công thức

* **Chuẩn hoá Z-score:** ( X' = (X - \mu)/\sigma ) theo cột

* **Ridge (closed-form):**
  [
  \hat{w} = (X^\top X + \alpha I)^{-1} X^\top y
  ]
  (Thêm bias qua cột 1; Ridge giúp ổn định khi đa cộng tuyến)

* **Biến đổi mục tiêu:** học trên ( t = \log(1+y) ), suy đoán ngược ( \hat{y} = \exp(\hat{t}) - 1 )

* **Đánh giá:**

  * ( \text{RMSE} = \sqrt{\frac{1}{n}\sum (y-\hat{y})^2} )
  * ( \text{MAE} = \frac{1}{n}\sum |y-\hat{y}| )
  * ( R^2 = 1 - \frac{\sum (y-\hat{y})^2}{\sum (y-\bar{y})^2} )

* **Cross-Validation:** K-Fold (tuỳ chọn **stratified theo bin** của (y) cho hồi quy)

**Implement bằng NumPy**

* Toàn bộ thao tác vector hoá bằng NumPy: chuẩn hoá, one-hot, KMeans mini, closed-form Ridge, K-Fold

---

## Installation & Setup

Yêu cầu Python ≥ 3.10

```bash
# Tạo môi trường (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
matplotlib
seaborn
```

Dữ liệu: load `AB_NYC_2019.csv` vào `data/raw/` (tải bằng Kaggle CLI hoặc tải thủ công tại [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data))

---

## Usage

Thứ tự chạy notebook:

1. **`01_data_exploration.ipynb`**

   * Load dữ liệu, thống kê thiếu/unique, mô tả cơ bản, đồ thị phân phối & geo scatter

2. **`02_preprocessing.ipynb`**

   * Tiền xử lý + tạo đặc trưng (bao gồm **neighbourhood** và **geo KMeans**)
   * Lưu ra `data/processed/ab_nyc_2019_processed_v2_geo.npz`

3. **`03_modeling.ipynb`**

   * Chia tập train/val/test; baseline trung bình
   * **K-Fold CV** chọn `alpha` (Ridge) **với log-transform mục tiêu**
   * Huấn luyện final (train+val), đánh giá test; trực quan hoá (pred vs. true, residuals)
   * (Tuỳ chọn) Lưu model `.npz`

> Code notebook gọi trực tiếp các hàm (API) trong `src/`

---

## Results

**Baseline tuyến tính (trước nâng cấp):**

* RMSE ≈ **103.91**, MAE ≈ **61.29**, R² ≈ **0.289**

**Sau nâng cấp (log-price + geo features + CV):**

* RMSE ≈ **98.18**, MAE ≈ **51.15**, R² ≈ **0.344**

**Nhận xét:**

* Đã cải thiện rõ (đặc biệt **MAE**).
* Sai số vẫn cao với định giá thực tế

**Trực quan hoá (trong notebook):**

* Histogram `price` trước/sau cleaning, phân vị
* Heatmap độ tương quan với `price`
* Scatter **Pred vs True**, histogram  cho residual

---

## Project Structure

```
.
├── data/
│   ├── raw/                      # đặt AB_NYC_2019.csv ở đây
│   └── processed/
│       └── ab_nyc_2019_processed_v2_geo.npz
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py               # re-export API
│   ├── data_processing.py        # load CSV RFC, cleaning, one-hot, KMeans(NumPy), assemble features, save/load .npz
│   ├── visualization.py          # histogram, bar, scatter geo, corr heatmap, pred-vs-true, residuals
│   └── models.py                 # split, standardize, Ridge closed-form, CV (K-Fold/stratified), log-target pipeline
├── requirements.txt
└── README.md
```

**Tóm chức năng file:**

* `data_processing.py`: đọc CSV (DictReader), lọc geo, impute, clip outlier, mã hoá phân loại, **KMeans NumPy**, lắp `X,y`, lưu `.npz`
* `visualization.py`: tiện ích vẽ Matplotlib (EDA + diagnostics)
* `models.py`: chia tập; chuẩn hoá; Ridge (closed-form); **pipeline có log-transform**; K-Fold CV; đánh giá

---

## Challenges & Solutions

1. **CSV có dấu phẩy trong chuỗi** ⇒ `genfromtxt` lệch cột
   **Giải pháp:** dùng `csv.DictReader` (quote-aware), cast về NumPy

2. **Phân phối giá lệch phải, ngoại lai lớn** kéo RMSE
   **Giải pháp:** clip percentiles + **log1p(target)** khi học

3. **Cardinality cao ở `neighbourhood`** → one-hot to, dễ nhiễu
   **Giải pháp:** gộp nhãn hiếm; (định hướng) dùng OOF target-encoding

4. **Vị trí phi tuyến mạnh** (lat/lon không tuyến tính)
   **Giải pháp:** **KMeans geo** + khoảng cách centroid; (định hướng) thêm **khoảng cách tới các vị trí trung tâm (ở đây lấy Time Squares)**

5. **Đánh giá có thể lạc quan theo không gian** (random split)
   **Giải pháp:** (định hướng) bổ sung **Group/Spatial CV** để kiểm tra tổng quát hoá theo không gian

---

## Future Improvements

* **Đặc trưng vị trí:** tương tác `room_type × geo`
* **Mã hoá phân loại:** **OOF target-encoding** cho `neighbourhood`/`host_id`
* **Mô hình:** RandomForest/XGBoost/LightGBM để bắt phi tuyến (đối chiếu với Ridge)
* **Text features:** n-gram từ `name` (bag-of-words thưa), sentiment/keyword
* **Đánh giá:** thêm **spatial/block CV** (group theo `neighbourhood` hoặc lưới lat-lon)
* **Chuẩn hoá quy trình:** script CLI tái lập end-to-end

---

## Contributors

* **Tác giả:** *Nguyễn Ngọc Khoa*
* **Contact:** *[23122036@student.hcmus.edu.vn](mailto:23122036@student.hcmus.edu.vn)* 

---

## License

**MIT License** — xem file `LICENSE`