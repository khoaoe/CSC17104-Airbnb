File này mô tả dữ liệu đã xử lý.
Để tạo airbnb_2019_preprocessed.npz thật, chạy notebook 02_preprocessing (đã bổ sung cell) hoặc:
    python scripts/make_preprocessed.py

Đầu ra: data/processed/airbnb_2019_preprocessed.npz
  - X: đặc trưng NumPy-only (đã one-hot phòng & nhóm khu vực + numeric clean)
  - y: giá (price)
  - room_type: danh sách nhãn cột one-hot phòng
  - neighbourhood_group: danh sách nhãn cột one-hot khu vực
