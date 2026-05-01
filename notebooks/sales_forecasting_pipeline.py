import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("1. DATA LOADING")
train_df = pd.read_csv('../data/Analytical/sales.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df = train_df.sort_values('Date').reset_index(drop=True)

test_df = pd.read_csv('../data/Operational/sample_submission.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df = test_df.sort_values('Date').reset_index(drop=True)

print("2. BUILDING ROBUST CALENDAR FEATURES (NO LAGS, NO TREND)")
vn_holidays = holidays.VN(years=range(2012, 2026))

# Target Encoding Dictionaries from Train Data
train_df['DayOfWeek'] = train_df['Date'].dt.dayofweek
train_df['Month'] = train_df['Date'].dt.month
dow_te_rev = train_df.groupby('DayOfWeek')['Revenue'].mean().to_dict()
month_te_rev = train_df.groupby('Month')['Revenue'].mean().to_dict()
dow_te_cogs = train_df.groupby('DayOfWeek')['COGS'].mean().to_dict()
month_te_cogs = train_df.groupby('Month')['COGS'].mean().to_dict()

mean_rev = train_df['Revenue'].mean()
mean_cogs = train_df['COGS'].mean()

def create_features(df, is_train=True):
    df_feat = pd.DataFrame()
    
    # Basic Time
    df_feat['Month'] = df['Date'].dt.month
    df_feat['Day'] = df['Date'].dt.day
    df_feat['DayOfWeek'] = df['Date'].dt.dayofweek
    df_feat['DayOfYear'] = df['Date'].dt.dayofyear
    df_feat['Quarter'] = df['Date'].dt.quarter
    
    # Binary
    df_feat['Is_Weekend'] = np.where(df_feat['DayOfWeek'] >= 5, 1, 0)
    df_feat['Is_Month_Start'] = df['Date'].dt.is_month_start.astype(int)
    df_feat['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
    df_feat['Is_Holiday'] = df['Date'].apply(lambda x: 1 if x in vn_holidays else 0)
    
    # Fourier Terms for smooth seasonality
    df_feat['sin_365'] = np.sin(2 * np.pi * df_feat['DayOfYear'] / 365.25)
    df_feat['cos_365'] = np.cos(2 * np.pi * df_feat['DayOfYear'] / 365.25)
    df_feat['sin_30'] = np.sin(2 * np.pi * df_feat['Day'] / 30.5)
    df_feat['cos_30'] = np.cos(2 * np.pi * df_feat['Day'] / 30.5)
    df_feat['sin_7'] = np.sin(2 * np.pi * df_feat['DayOfWeek'] / 7.0)
    df_feat['cos_7'] = np.cos(2 * np.pi * df_feat['DayOfWeek'] / 7.0)
    
    # Target Encoding
    df_feat['TE_DoW_Rev'] = df_feat['DayOfWeek'].map(dow_te_rev).fillna(mean_rev)
    df_feat['TE_Month_Rev'] = df_feat['Month'].map(month_te_rev).fillna(mean_rev)
    df_feat['TE_DoW_COGS'] = df_feat['DayOfWeek'].map(dow_te_cogs).fillna(mean_cogs)
    df_feat['TE_Month_COGS'] = df_feat['Month'].map(month_te_cogs).fillna(mean_cogs)
    
    return df_feat

X_train = create_features(train_df)
y_train_rev = train_df['Revenue']
y_train_cogs = train_df['COGS']

X_test = create_features(test_df, is_train=False)

print("3. TRAINING ADVANCED ENSEMBLE (LightGBM + XGBoost)")
# Mô hình 1: LightGBM (Tối ưu độ sâu để tránh học thuộc, học rate thấp để hội tụ mượt)
lgb_rev = LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
lgb_cogs = LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)

# Mô hình 2: XGBoost (Mạnh mẽ trong việc bắt các ngoại lệ/đột biến Lễ Tết)
xgb_rev = XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_cogs = XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

print("  -> Training Revenue Models...")
lgb_rev.fit(X_train, y_train_rev)
xgb_rev.fit(X_train, y_train_rev)

print("  -> Training COGS Models...")
lgb_cogs.fit(X_train, y_train_cogs)
xgb_cogs.fit(X_train, y_train_cogs)

print("4. DIRECT FORECASTING (1-STEP MULTI-HORIZON)")
pred_lgb_rev = lgb_rev.predict(X_test)
pred_xgb_rev = xgb_rev.predict(X_test)

pred_lgb_cogs = lgb_cogs.predict(X_test)
pred_xgb_cogs = xgb_cogs.predict(X_test)

# Lấy trung bình cộng (Ensemble)
final_pred_rev = (pred_lgb_rev + pred_xgb_rev) / 2.0
final_pred_cogs = (pred_lgb_cogs + pred_xgb_cogs) / 2.0

# Đảm bảo không có giá trị âm
final_pred_rev = np.maximum(final_pred_rev, 0)
final_pred_cogs = np.maximum(final_pred_cogs, 0)

print("5. SAVING SUBMISSION")
submission = test_df.copy()
submission['Revenue'] = final_pred_rev
submission['COGS'] = final_pred_cogs
submission.to_csv('../output/submission.csv', index=False)
print("Saved Direct Ensemble predictions to 'submission.csv'")

# Đồ thị biểu diễn
plt.figure(figsize=(15, 6))
plt.plot(train_df['Date'], train_df['Revenue'], label='Actual Revenue (Train)', lw=1)
plt.plot(submission['Date'], submission['Revenue'], label='Predicted Revenue (Test - Direct ML)', lw=2, alpha=0.8)
plt.title('Revenue Forecasting (Direct Machine Learning Ensemble)')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.tight_layout()
plt.savefig('../output/direct_ml_extrapolation.png')
plt.close()
print("Pipeline complete!")
