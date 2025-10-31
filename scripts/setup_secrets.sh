#!/usr/bin/env bash
set -euo pipefail
# Đọc biến từ .env nếu có (định dạng KEY=VALUE, không có dấu nháy)
if [ -f ".env" ]; then
  # shellcheck disable=SC2046
  export $(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env | xargs)
fi

# Tạo kaggle.json từ env nếu người dùng không đặt sẵn tại ~/.kaggle/kaggle.json
KAGGLE_DIR="${HOME}/.kaggle"
KAGGLE_JSON="${KAGGLE_DIR}/kaggle.json"
mkdir -p "${KAGGLE_DIR}"
if [ ! -f "${KAGGLE_JSON}" ] && [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  # Kaggle API yêu cầu file JSON đúng schema {username, key}
  cat > "${KAGGLE_JSON}" <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
  chmod 600 "${KAGGLE_JSON}"
  echo "[INFO] Tạo ${KAGGLE_JSON} từ .env"
fi

# Nếu vẫn không có file và không có biến, nhắc người dùng
if [ ! -f "${KAGGLE_JSON}" ] && { [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; }; then
  echo "[ERROR] Thiếu thông tin Kaggle. Tạo ~/.kaggle/kaggle.json hoặc khai báo KAGGLE_USERNAME/KAGGLE_KEY trong .env" >&2
  exit 1
fi
echo "[OK] Secrets sẵn sàng (Kaggle API)."