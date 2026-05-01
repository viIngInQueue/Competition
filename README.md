# Competition — Phân Tích, Trực Quan Và Xây Dựng Mô Hình Dự Đoán Doanh Thu


## 🎯 Tổng quan

Phân tích dữ liệu, tạo dashboard và xây dựng mô hình học máy cho bài toán dự đoán doanh thu tương lai của 1 công ti thương mại điện tử tại Việt Nam
với tập dữ liệu cho việc phân tích từ năm 2012-2022

**Mô hình:**
- LightGBM + XGBoost ensemble  
---

## 📂 Cấu trúc Thư mục

```
Competition/
├── README.md                                    # Tài liệu này
├── requirements.txt                             # Dependencies
├── data/
│   ├── Analytical/  → sales.csv (training data)
│   ├── Master/      → customers, products, geography, promotions  
│   ├── Operational/ → inventory, sample_submission, web_traffic
│   └── Transaction/ → orders, payments, returns, reviews, shipments
├── notebooks/
│   ├── Analysis.ipynb                           # EDA
│   ├── sales_forecasting_pipeline.ipynb         # v1 (Original)
│   ├── MCQ.ipynb
│   └── dashboard.html
└── output/
    └── submission.csv                           # Final predictions
    └── direct_ml_extrapolation.png
```

---

## 🔧 Thiết lập Môi trường

### Option 1: Virtual Environment (Recommended)

```powershell
# Tạo & kích hoạt
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Anaconda

```powershell
conda create -n competition python=3.10 -y
conda activate competition
pip install -r requirements.txt
```

---

## 🚀 Chạy Pipeline

### ⭐ Chạy model:
```powershell
cd notebooks
jupyter notebook sales_forecasting_pipeline.ipynb
```




