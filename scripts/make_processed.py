#!/usr/bin/env python3
from pathlib import Path
import argparse
from src.data_processing import preprocess_and_save, RAW_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Đường dẫn CSV trong data/raw (mặc định: file .csv đầu tiên)")
    ap.add_argument("--out", type=str, default=str(Path("data/processed/airbnb_processed.npz")))
    args = ap.parse_args()

    if args.csv is None:
        # chọn file csv đầu tiên trong RAW_DIR
        csvs = sorted(RAW_DIR.glob("*.csv")) + sorted(RAW_DIR.glob("*.csv.gz"))
        if not csvs:
            raise SystemExit(f"Không tìm thấy CSV trong {RAW_DIR}")
        csv_path = csvs[0]
    else:
        csv_path = Path(args.csv)

    preprocess_and_save(csv_path, Path(args.out))

if __name__ == "__main__":
    main()
