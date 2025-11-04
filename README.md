# [CSC17104] Homework 02 — NYC Airbnb (NumPy-only)

## 1) Mô tả ngắn

Mini-project dùng **NumPy thuần** để khám phá và tiền xử lý bộ dữ liệu **New York City Airbnb Open Data**, modeling để dự đoán giá.


## 2) Mục lục

* [Giới thiệu](#giới-thiệu)
* [Dataset](#dataset)
* [Method (NumPy-only)](#method-numpy-only)
* [Installation & Setup](#installation--setup)
* [Usage](#usage)
* [Kết quả](#kết-quả)
* [Project Structure](#project-structure)
* [Challenges & Solutions](#challenges--solutions)
* [Future Improvements](#future-improvements)
* [Contributors & License](#contributors--license)


## Giới thiệu

Bài toán: phân tích dữ liệu lưu trú ngắn hạn (Airbnb) tại NYC để nhìn các khuynh hướng cơ bản và chuẩn bị đặc trưng cho bước mô hình hoá tối giản. Bài tập nhấn mạnh:

* Xử lý/biến đổi dữ liệu **chỉ với NumPy**; không dùng pandas cho pipeline.
* Trực quan hoá bằng **Matplotlib/Seaborn** (ở 01).
* (Tuỳ chọn nâng cao/bonus) hiện thực một số thuật toán ML bằng NumPy ở 03.


## Dataset

* Nguồn: Kaggle — *New York City Airbnb Open Data*.
  Link: [https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
* Các cột quan trọng: `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`, `neighbourhood_group`, `room_type`, `latitude`, `longitude`.


## Method (NumPy-only)

### 01_data_exploration

* Đọc CSV → NumPy (không pandas), thống kê mô tả numeric & tần suất categorical; kiểm tra thiếu, phạm vi hợp lệ; in top nhóm phổ biến.

### 02_preprocessing

* Numeric: điền `reviews_per_month = 0`, các cột số khác dùng median.
* Clip ngoại lai theo percentile; chuẩn hoá **Z-score**.
* One-hot `neighbourhood_group`, `room_type`.
* Biến mục tiêu: `y = log1p(price)`.
* Lưu `.npz`: `data/processed/airbnb_2019_preprocessed.npz`.


## Installation & Setup

```bash
# (tuỳ chọn) tạo venv
python3 -m venv .venv && source .venv/bin/activate

# cài thư viện
pip install -r requirements.txt
```

Thiết lập Kaggle API (nếu cần tải dữ liệu gốc): tạo `~/.kaggle/kaggle.json` hoặc `.env` chứa `KAGGLE_USERNAME`, `KAGGLE_KEY`.


## Usage

### 1) Khám phá dữ liệu

Mở `notebooks/01_data_exploration.ipynb` và chạy tuần tự.

### 2) Tiền xử lý (tạo .npz)

Mở `notebooks/02_preprocessing.ipynb` và chạy đến cell cuối.

```python
from pathlib import Path
from src.data_processing import preprocess_and_save
preprocess_and_save(Path("data/raw/AB_NYC_2019.csv"))
```

File đầu ra: `data/processed/airbnb_2019_preprocessed.npz`


## Kết quả 

* **Exploration**: đã in đầy đủ thống kê numeric, tần suất categorical (Manhattan/Brooklyn...; Entire home/apt/Private room/Shared room).
* **Preprocessing**: sinh `X` (numeric clean + z-score + one-hot) và `y = log1p(price)`; sanity check: không NaN/Inf.


## Project Structure

```
project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
└── src/
    ├── __init__.py
    ├── data_processing.py
    ├── visualization.py
    └── models.py
```

## Challenges & Solutions

* **Thiếu giá trị & ngoại lai**: `reviews_per_month` trống nhiều → điền 0; cột khác dùng median; clip P1–P99.
* **Chuẩn hoá đặc trưng**: Z-score cho toàn bộ X; giải pháp giảm ảnh hưởng tính toán không đồng thang.

## Future Improvements



## Contributors & License

* **Tác giả**: Nguyễn Ngọc Khoa, 23122036, 23TNT1
* **Liên hệ**: 23122036@student.hcmus.edu.vn
* **License**: MIT 