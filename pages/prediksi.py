import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Prediksi Penjualan",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background:#E9EEF5;
}
/* Sidebar menu text */
.menu-text{
    color:white;
    font-size:22px;
    font-weight:700;
    margin-top:18px;
}

/* Button icon sidebar */
.stButton > button{
    background:#2A245D !important;
    border:none !important;
    border-radius:18px !important;
    height:70px !important;
    width:70px !important;
    font-size:35px !important;
    color:white !important;
    box-shadow:none !important;
}

.stButton > button:hover{
    background:#40358C !important;
}
/* Title */
.main-title{
    font-size:42px;
    font-weight:800;
    color:#111827;
}

.sub-title{
    color:#64748B;
    font-size:18px;
}

/* KPI */
.metric-card{
    background:#FFFFFF;
    border-radius:20px;
    padding:20px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    border:1px solid #DCE3EC;
    text-align:center;
    height:140px;
}
/* Label KPI */
[data-testid="stMetricLabel"]{
    color:#64748B !important;
    font-size:100px !important;
    font-weight:1000 !important;
}
            /* Default angka KPI */
[data-testid="stMetricValue"]{
    color:#0F172A !important;
    font-weight:700 !important;
}

.metric-title{
    color:#64748B;
    font-size:18px;
    font-weight:700;
}

.metric-value{
    color:#0F172A;
    font-size:34px;
    font-weight:800;
    margin-top:10px;
}

/* Card chart */
.chart-card{
    background:#FFFFFF;
    border-radius:22px;
    padding:20px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}

/* Filter */
.filter-title{
    font-size:28px;
    font-weight:800;
    color:#0F172A;
}
/* Label filter jadi hitam */
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] p {
    color:#111827 !important;
    font-weight:700 !important;
    font-size:16px !important;
    opacity:1 !important;
}
/* Paksa semua text filter jadi hitam */
label{
    color:#111827 !important;
}

/* Box select */
.stSelectbox div[data-baseweb="select"]{
    background:#222531 !important;
    border-radius:14px !important;
    border:1px solid #222531 !important;
}

/* Tulisan dalam dropdown */
.stSelectbox div{
    color:white !important;
}

/* Dropdown menu */
div[role="listbox"]{
    background:#222531 !important;
    color:white !important;
}

div[role="option"]{
    color:white !important;
    background:#222531 !important;
}

div[role="option"]:hover{
    background:#394150 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    "data/AyamSerayu_3Years_Transaction_Data.csv"
)
df.columns = df.columns.str.strip()

df["Tanggal & Waktu"] = pd.to_datetime(
    df["Tanggal & Waktu"],
    errors="coerce"
)

df["Tanggal"] = (
    df["Tanggal & Waktu"]
    .dt.date
)
df["Tanggal"] = pd.to_datetime(df["Tanggal"])

df["tahun"] = df["Tanggal"].dt.year
df["bulan"] = df["Tanggal"].dt.month
# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class='main-title'>
PREDIKSI PENJUALAN<br>
PRODUK UMKM AYAM SERAYU
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub-title'>
Analisis data historis penjualan dan visualisasi interaktif
</div>
""", unsafe_allow_html=True)

st.write("")
# =========================================================
# TITLE
# =========================================================
with st.sidebar:

    st.markdown("""
    <div class="sidebar-title">
        AYAM SERAYU
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Dashboard
    col1, col2 = st.columns([1,4])

    with col1:
        if st.button("🏠", key="dashboard_btn"):
            st.switch_page("pages/dashboard.py")

    with col2:
        st.markdown(
            "<div class='menu-text'>Dashboard</div>",
            unsafe_allow_html=True
        )

    st.write("")

    # Prediksi
    col3, col4 = st.columns([1,4])

    with col3:
        if st.button("🔮", key="prediksi_btn"):
            st.switch_page("pages/prediksi.py")

    with col4:
        st.markdown(
            "<div class='menu-text'>Prediksi</div>",
            unsafe_allow_html=True
        )

# =========================================================
# FILTER (HIDDEN LOGIC)
# =========================================================

# state filter
outlet = st.session_state.get(
    "outlet_prediksi",
    "Semua"
)

produk = st.session_state.get(
    "produk_prediksi",
    "Semua"
)

horizon = st.session_state.get(
    "hari_prediksi",
    7
)

# filter dataframe
filtered_df = df.copy()

if outlet != "Semua":
    filtered_df = filtered_df[
        filtered_df["Outlet"] == outlet
    ]

if produk != "Semua":
    filtered_df = filtered_df[
        filtered_df["Nama Produk"] == produk
    ]

# =========================================================
# DATA KHUSUS GRAFIK
# =========================================================

daily_chart = (
    filtered_df
    .groupby("Tanggal")["Jumlah Produk"]
    .sum()
    .reset_index()
)

daily_chart = daily_chart.sort_values(
    "Tanggal"
)

daily_chart["lag_1"] = (
    daily_chart["Jumlah Produk"]
    .shift(1)
)

daily_chart["lag_7"] = (
    daily_chart["Jumlah Produk"]
    .shift(7)
)

daily_chart["rolling_mean_7"] = (
    daily_chart["Jumlah Produk"]
    .rolling(7)
    .mean()
)

daily_chart["rolling_std_7"] = (
    daily_chart["Jumlah Produk"]
    .rolling(7)
    .std()
)

daily_chart = daily_chart.dropna()

# =========================================================
# AGREGASI HARIAN
# =========================================================

daily = (
    df
    .groupby("Tanggal")["Jumlah Produk"]
    .sum()
    .reset_index()
)

daily = daily.sort_values("Tanggal")


# =========================================================
# FEATURE ENGINEERING
# =========================================================

daily["lag_1"] = (
    daily["Jumlah Produk"]
    .shift(1)
)

daily["lag_7"] = (
    daily["Jumlah Produk"]
    .shift(7)
)

daily["rolling_mean_7"] = (
    daily["Jumlah Produk"]
    .rolling(7)
    .mean()
)

daily["rolling_std_7"] = (
    daily["Jumlah Produk"]
    .rolling(7)
    .std()
)

daily = daily.dropna()


# =========================================================
# TRAIN TEST
# =========================================================

train_size = int(len(daily) * 0.8)

train = daily[:train_size]
test = daily[train_size:]

features = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7"
]

X_train = train[features]
y_train = train["Jumlah Produk"]

X_test = test[features]
y_test = test["Jumlah Produk"]


# =========================================================
# MODEL XGBOOST
# =========================================================

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# =========================================================
# METRICS
# =========================================================

mape = np.mean(
    np.abs(
        (y_test - predictions)
        / y_test
    )
) * 100


# =========================================================
# FUTURE FORECAST
# =========================================================

future_preds = []

last_values = (
    daily["Jumlah Produk"]
    .tail(7)
    .tolist()
)

last_date = daily["Tanggal"].max()

for i in range(horizon):

    lag_1 = last_values[-1]
    lag_7 = last_values[-7]

    rolling_mean = np.mean(
        last_values[-7:]
    )

    rolling_std = np.std(
        last_values[-7:]
    )

    X_future = pd.DataFrame({
        "lag_1":[lag_1],
        "lag_7":[lag_7],
        "rolling_mean_7":[rolling_mean],
        "rolling_std_7":[rolling_std]
    })

    pred = model.predict(
        X_future
    )[0]

    future_preds.append(pred)
    last_values.append(pred)

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=horizon
)

future_df = pd.DataFrame({
    "Tanggal": future_dates,
    "Prediksi": future_preds
})

# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

mape = np.mean(
    np.abs(
        (y_test - predictions)
        / y_test
    )
) * 100
# =========================================================
# KPI (PALING ATAS)
# =========================================================

# =========================================================
# KPI
# =========================================================

trend = (
    "Naik 📈"
    if future_preds[-1]
    > future_preds[0]
    else "Turun 📉"
)

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)

with c1:
    st.metric(
        "Model",
        "XGBoost"
    )

with c2:
    st.metric(
        "MAPE",
        f"{mape:.2f}%"
    )

with c3:
    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

with c4:
    st.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

with c5:
    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )

with c6:
    st.metric(
        f"Prediksi {horizon} Hari",
        f"{sum(future_preds):,.0f}"
    )

with c7:
    st.metric(
        "Trend Prediksi",
        trend
    )

st.write("")

# =========================================================
# FILTER + CHART
# =========================================================

left_filter, right_chart = st.columns([1,3])

with left_filter:

    st.markdown(
        "<div class='filter-title'>Filter Prediksi</div>",
        unsafe_allow_html=True
    )

    st.selectbox(
        "Filter Outlet",
        ["Semua"] +
        sorted(
            df["Outlet"]
            .dropna()
            .unique()
        ),
        key="outlet_prediksi"
    )

    st.selectbox(
        "Filter Produk",
        ["Semua"] +
        sorted(
            df["Nama Produk"]
            .dropna()
            .unique()
        ),
        key="produk_prediksi"
    )

    horizon = st.number_input(
        "Jumlah Hari Prediksi",
        min_value=1,
        max_value=365,
        value=14,
        step=1,
        key="hari_prediksi"
    )

# =========================================================
# MODEL KHUSUS GRAFIK
# =========================================================

if len(daily_chart) > 30:

    train_size_chart = int(
        len(daily_chart) * 0.8
    )

    train_chart = daily_chart[
        :train_size_chart
    ]

    test_chart = daily_chart[
        train_size_chart:
    ]

    X_train_chart = train_chart[
        features
    ]

    y_train_chart = train_chart[
        "Jumlah Produk"
    ]

    X_test_chart = test_chart[
        features
    ]

    y_test_chart = test_chart[
        "Jumlah Produk"
    ]

    model_chart = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model_chart.fit(
        X_train_chart,
        y_train_chart
    )

    predictions_chart = (
        model_chart.predict(
            X_test_chart
        )
    )

future_preds_chart = []

last_values_chart = (
    daily_chart["Jumlah Produk"]
    .tail(7)
    .tolist()
)

last_date_chart = (
    daily_chart["Tanggal"]
    .max()
)

for i in range(horizon):

    lag_1 = last_values_chart[-1]
    lag_7 = last_values_chart[-7]

    rolling_mean = np.mean(
        last_values_chart[-7:]
    )

    rolling_std = np.std(
        last_values_chart[-7:]
    )

    X_future = pd.DataFrame({
        "lag_1":[lag_1],
        "lag_7":[lag_7],
        "rolling_mean_7":[rolling_mean],
        "rolling_std_7":[rolling_std]
    })

    pred = model_chart.predict(
        X_future
    )[0]

    future_preds_chart.append(pred)

    last_values_chart.append(pred)

future_dates_chart = pd.date_range(
    start=last_date_chart
    + pd.Timedelta(days=1),
    periods=horizon
)

future_df_chart = pd.DataFrame({
    "Tanggal": future_dates_chart,
    "Prediksi": future_preds_chart
})
# =========================================================
# GRAFIK
# =========================================================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=test_chart["Tanggal"],
        y=y_test_chart,
        mode="lines",
        name="Actual",
        line=dict(color="#2563EB")
    )
)

fig.add_trace(
    go.Scatter(
        x=test_chart["Tanggal"],
        y=predictions_chart,
        mode="lines",
        name="Predicted",
        line=dict(color="#16A34A")
    )
)

fig.add_trace(
    go.Scatter(
        x=future_df_chart["Tanggal"],
        y=future_df_chart["Prediksi"],
        mode="lines",
        name="Forecast",
        line=dict(color="#982222")
    )
)

fig.update_layout(
    title="Grafik Historical Actual vs Predict",
    paper_bgcolor="#E4EFF9",
    plot_bgcolor="#E4EFF9",

    font=dict(
        color="black",
        size=14
    ),

    title_font=dict(
        color="black",
        size=22
    ),

    xaxis=dict(
        title="Tanggal",
        tickfont=dict(color="black"),
        gridcolor="#030303"
    ),

    yaxis=dict(
        title="Jumlah Produk",
        tickfont=dict(color="black"),
        gridcolor="#0C0C0D"
    ),

    legend=dict(
        font=dict(
            color="black",
            size=13
        )
    ),

    height=500
)

with right_chart:
    st.plotly_chart(
        fig,
        use_container_width=True
    )
# =========================================================
# TABLE
# =========================================================

st.subheader(
    "Tabel Hasil Prediksi"
)

st.dataframe(
    future_df_chart,
    use_container_width=True
)

csv = future_df_chart.to_csv(
    index=False
)

st.download_button(
    "Download CSV",
    csv,
    "hasil_prediksi.csv",
    "text/csv"
)