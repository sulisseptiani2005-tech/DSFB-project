# =========================================================
# 1. IMPORT LIBRARY
# =========================================================

import os
import webbrowser
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# 2. MEMBUAT FOLDER VISUALISASI
# =========================================================

folder_visualisasi = "visualisasi"

if not os.path.exists(folder_visualisasi):
    os.makedirs(folder_visualisasi)

print("Folder visualisasi siap")

# =========================================================
# 3. LOAD DATASET
# =========================================================

df = pd.read_csv(
    "data/AyamSerayu_3Years_Transaction_Data.csv"
)

print("\n===== DATASET BERHASIL DIMUAT =====")
print(df.head())

# =========================================================
# 4. DATA UNDERSTANDING
# =========================================================

print("\n===== INFORMASI DATA =====")
print("Jumlah Baris :", df.shape[0])
print("Jumlah Kolom :", df.shape[1])

print("\n===== MISSING VALUE =====")
print(df.isnull().sum())

produk_laris = (
    df.groupby("Nama Produk")["Jumlah Produk"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== 20 PRODUK TERLARIS =====")
print(produk_laris.head(20))

# =========================================================
# 5. DATA PREPROCESSING
# =========================================================

df.columns = df.columns.str.strip()

df = df[[
    "Tanggal & Waktu",
    "Outlet",
    "Nama Produk",
    "Jumlah Produk"
]]

print("\n===== DATA AWAL =====")
print("Shape awal:", df.shape)

print("\n===== DUPLICATE DATA =====")
print(df.duplicated().sum())

df = df.drop_duplicates()

df["Tanggal & Waktu"] = pd.to_datetime(
    df["Tanggal & Waktu"],
    errors="coerce"
)

df = df.dropna(
    subset=["Tanggal & Waktu"]
)

df = df.sort_values(
    "Tanggal & Waktu"
).reset_index(drop=True)

print("\nShape final:", df.shape)

# =========================================================
# 6. TIME FEATURE EXTRACTION
# =========================================================

df["Tanggal"] = (
    df["Tanggal & Waktu"]
    .dt.date
)

df["Tanggal"] = pd.to_datetime(
    df["Tanggal"]
)

df["tahun"] = (
    df["Tanggal"].dt.year
)

df["bulan"] = (
    df["Tanggal"].dt.month
)

df["hari"] = (
    df["Tanggal"].dt.day
)

df["hari_ke"] = (
    df["Tanggal"].dt.dayofweek
)

# =========================================================
# 7. DAILY SALES AGGREGATION
# =========================================================

df_daily = (
    df.groupby(
        "Tanggal",
        as_index=False
    )["Jumlah Produk"]
    .sum()
    .rename(
        columns={
            "Jumlah Produk": "target"
        }
    )
)

df_daily["Tanggal"] = pd.to_datetime(
    df_daily["Tanggal"]
)

# =========================================================
# 8. MONTHLY SALES
# =========================================================

df_daily["tahun"] = (
    df_daily["Tanggal"].dt.year
)

df_daily["bulan"] = (
    df_daily["Tanggal"].dt.month
)
avg_bulanan = (
    df_daily.groupby("bulan")
    ["target"]
    .mean()
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=avg_bulanan.index,
    y=avg_bulanan.values,
    mode="lines+markers",
    name="Average Sales"
))

fig.update_layout(
    title="Rata-rata Penjualan Harian per Bulan",
    xaxis_title="Bulan",
    yaxis_title="Rata-rata Penjualan",
    template="plotly_white"
)

path = os.path.join(
    folder_visualisasi,
    "monthly_sales.html"
)

fig.write_html(path)

# =========================================================
# 9. YEARLY SALES
# =========================================================

avg_tahunan = (
    df_daily.groupby("tahun")
    ["target"]
    .mean()
    .reset_index()
)

fig = px.bar(
    avg_tahunan,
    x="tahun",
    y="target",
    text="target"
)

fig.update_layout(
    title="Rata-rata Penjualan Tahunan",
    template="plotly_white"
)

path = os.path.join(
    folder_visualisasi,
    "yearly_sales.html"
)

fig.write_html(path)

# =========================================================
# 10. SALES TREND
# =========================================================

df_model = (
    df.groupby(
        "Tanggal",
        as_index=False
    )["Jumlah Produk"]
    .sum()
    .rename(
        columns={
            "Jumlah Produk": "target"
        }
    )
)

df_model["Tanggal"] = pd.to_datetime(
    df_model["Tanggal"]
)

df_model = (
    df_model.sort_values(
        "Tanggal"
    )
    .reset_index(drop=True)
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_model["Tanggal"],
    y=df_model["target"],
    mode="lines",
    name="Sales Trend"
))

fig.update_layout(
    title="Sales Trend Over Time",
    xaxis_title="Tanggal",
    yaxis_title="Jumlah Penjualan",
    template="plotly_white"
)

path = os.path.join(
    folder_visualisasi,
    "sales_trend.html"
)

fig.write_html(path)

# =========================================================
# 11. FEATURE ENGINEERING
# =========================================================

def create_features(data):

    df_feat = data.copy()

    df_feat["day"] = (
        df_feat["Tanggal"].dt.day
    )

    df_feat["month"] = (
        df_feat["Tanggal"].dt.month
    )

    df_feat["dayofweek"] = (
        df_feat["Tanggal"].dt.dayofweek
    )

    df_feat["quarter"] = (
        df_feat["Tanggal"].dt.quarter
    )

    df_feat["lag_1"] = (
        df_feat["target"].shift(1)
    )

    df_feat["lag_7"] = (
        df_feat["target"].shift(7)
    )

    df_feat["lag_14"] = (
        df_feat["target"].shift(14)
    )

    df_feat["rolling_mean_7"] = (
        df_feat["target"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df_feat["rolling_mean_14"] = (
        df_feat["target"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    df_feat["rolling_std_7"] = (
        df_feat["target"]
        .shift(1)
        .rolling(7)
        .std()
    )

    return df_feat


df_feat = create_features(
    df_model
)

df_feat = (
    df_feat.dropna()
    .reset_index(drop=True)
)

# =========================================================
# 12. TRAIN TEST SPLIT
# =========================================================

features = [
    "day", "month",
    "dayofweek", "quarter",
    "lag_1", "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7"
]

split = int(
    len(df_feat) * 0.8
)

train = df_feat.iloc[:split]
test = df_feat.iloc[split:]

X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]

# =========================================================
# 13. MODEL TRAINING
# =========================================================

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# =========================================================
# 14. MODEL EVALUATION
# =========================================================

pred = model.predict(X_test)

print("\n===== MODEL EVALUATION =====")

print(
    "MAE:",
    mean_absolute_error(
        y_test,
        pred
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )
)

print(
    "R2:",
    r2_score(
        y_test,
        pred
    )
)

# =========================================================
# 15. ACTUAL VS PREDICTED
# =========================================================

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=test["Tanggal"],
    y=y_test,
    mode="lines",
    name="Actual"
))

fig.add_trace(go.Scatter(
    x=test["Tanggal"],
    y=pred,
    mode="lines",
    name="Predicted"
))

fig.update_layout(
    title="Actual vs Predicted",
    template="plotly_white"
)

path = os.path.join(
    folder_visualisasi,
    "actual_vs_predicted.html"
)

fig.write_html(path)

# =========================================================
# 16. AUTO OPEN
# =========================================================

for file in os.listdir(folder_visualisasi):

    if file.endswith(".html"):

        webbrowser.open(
            os.path.abspath(
                os.path.join(
                    folder_visualisasi,
                    file
                )
            )
        )

print("\n===== SELESAI =====")
print("Semua visualisasi tersimpan di folder visualisasi")