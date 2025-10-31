## Secrets (.env) & Kaggle API
- Tạo secrets bằng `.env`: copy `.env.example` -> đổi tên thành `.env` và điền `KAGGLE_USERNAME`, `KAGGLE_KEY`.
- Chạy thiết lập:
  - Linux/macOS: `bash scripts/setup_secrets.sh`
  - Windows (PowerShell): `./scripts/setup_secrets.ps1`
- Ưu tiên `~/.kaggle/kaggle.json`; nếu chưa có, script sẽ tạo từ `.env`.
- Tham khảo: Kaggle API yêu cầu `~/.kaggle/kaggle.json` hoặc biến môi trường `KAGGLE_USERNAME`/`KAGGLE_KEY`.