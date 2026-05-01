# Competition — Dự báo Doanh số & Chi phí Bán hàng

Kho chứa dữ liệu, notebook và tài nguyên cho bài toán dự báo Revenue (Doanh số) và COGS (Chi phí hàng bán) bằng Machine Learning.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Thiết lập môi trường](#thiết-lập-môi-trường)
- [Chạy Pipeline](#chạy-pipeline)
- [Các phiên bản](#các-phiên-bản)
- [Kết quả](#kết-quả)

---

## 🎯 Tổng quan

Dự án sử dụng dữ liệu bán hàng để xây dựng mô hình dự báo:
- **Revenue** (Doanh số): Tổng doanh thu
- **COGS** (Chi phí hàng bán): Chi phí sản xuất/nhập

**Phương pháp v2 (Recommended):**
- Advanced feature engineering: Calendar, Lag, Rolling statistics
- LightGBM + XGBoost ensemble  
- TimeSeriesSplit cross-validation
- Weighted ensemble dạo trên hiệu suất CV

---

## 📂 Cấu trúc Thư mục

```
Competition/
├── README.md                                    # Tài liệu này
├── IMPROVEMENTS.md                              # Chi tiết cải tiến
├── requirements.txt                             # Dependencies
├── data/
│   ├── Analytical/  → sales.csv (training data)
│   ├── Master/      → customers, products, geography, promotions  
│   ├── Operational/ → inventory, sample_submission, web_traffic
│   └── Transaction/ → orders, payments, returns, reviews, shipments
├── notebooks/
│   ├── Analysis.ipynb                           # EDA
│   ├── sales_forecasting_pipeline.ipynb         # v1 (Original)
│   ├── sales_forecasting_pipeline_v2.ipynb      # v2 (Recommended ⭐)
│   ├── MCQ.ipynb
│   └── dashboard.html
└── output/
    └── submission.csv                           # Final predictions
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

### ⭐ Chạy v2 (Recommended):
```powershell
cd notebooks
jupyter notebook sales_forecasting_pipeline_v2.ipynb
```

### Chạy v1 (Original):
```powershell
jupyter notebook sales_forecasting_pipeline.ipynb
```

---

## 🔄 Các Phiên Bản

### v2 (Recommended) ⭐⭐⭐

**Cải tiến:**
- ✅ Advanced features: Lag (1,3,7,14,30) + Rolling (7,14,30)
- ✅ Feature selection: SelectKBest top 30
- ✅ Early stopping: Validation-based
- ✅ TimeSeriesSplit: Proper time-series CV
- ✅ Weighted ensemble: Adaptive weights
- ✅ Retraining: Full data for final model

**Hiệu suất:** ~5-10% cải tiến so với v1

---

### v1 (Original)

**Tính năng:**
- Calendar + Target encoding
- LightGBM + XGBoost
- Optimized hyperparameters
- Simple average ensemble

---

## 📈 Kết Quả

Khi chạy v2, bạn sẽ nhận:

```
✓ 6. TIME-SERIES CROSS-VALIDATION
  [LGB Revenue] Mean MAE: xxxxxx.xx ±xxxx.xx
  [XGB Revenue] Mean MAE: xxxxxx.xx ±xxxx.xx
  [LGB COGS   ] Mean MAE: xxxxxx.xx ±xxxx.xx
  [XGB COGS   ] Mean MAE: xxxxxx.xx ±xxxx.xx

✓ 8. WEIGHTED ENSEMBLE
  Revenue: LGB=0.XXX, XGB=0.XXX
  COGS:    LGB=0.XXX, XGB=0.XXX

✓ 10. TRAIN METRICS
  Revenue → MAE: xxxxxxxxx.xx RMSE: xxxxxxxxx.xx MAPE: xxx.xx%
  COGS    → MAE: xxxxxxxxx.xx RMSE: xxxxxxxxx.xx MAPE: xxx.xx%

✓ 11. SAVING SUBMISSION
  output/submission.csv
```

---

## 📚 Tài liệu Thêm

- **IMPROVEMENTS.md** — Chi tiết technical của các cải tiến
- **requirements.txt** — Danh sách packages
- **notebooks/** — Code thực thi

---

## ✅ Checklist Trước Submit

- [ ] Chạy v2 thành công
- [ ] `output/submission.csv` tồn tại  
- [ ] Dự báo hợp lý (no NaN, reasonable range)
- [ ] CV metrics OK (MAE/RMSE)

---

**Cập nhật**: May 1, 2026 | **Version**: 2.0 | **Status**: ✅ Production-ready




