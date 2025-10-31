## Secrets (.env) & Kaggle API 

* Tạo secrets bằng `.env`: copy `.env.example` → đổi tên thành `.env` và điền `KAGGLE_USERNAME`, `KAGGLE_KEY`.
* Thiết lập trên Linux:

  ```bash
  bash scripts/setup_secrets.sh
  ```

* Ưu tiên `~/.kaggle/kaggle.json`; nếu chưa có, script sẽ tạo từ `.env`.
* Kaggle API cần `~/.kaggle/kaggle.json` **hoặc** biến môi trường `KAGGLE_USERNAME`/`KAGGLE_KEY`.

---

## Môi trường ảo

1. Cài venv (nếu chưa có):

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv
```

2. Tạo & kích hoạt môi trường ảo trong thư mục dự án:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Cài dependencies từ `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -U kaggle # kaggle-cli để tải dataset
chmod 600 ~/.kaggle/kaggle.json # cấp quyền
```

4. Thoát môi trường khi xong:

```bash
deactivate
```

**Ghi chú**

* Thêm `.venv/` vào `.gitignore`.
* Nếu cài thêm thư viện mới, cập nhật:

```bash
pip freeze > requirements.txt
```
