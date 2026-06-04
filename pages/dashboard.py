import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Dashboard UMKM Ayam Serayu",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

/* Hide Streamlit */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Background */
.stApp{
    background:#E9EEF5 !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#13103B;
    width:260px !important;
}

.sidebar-title{
    color:white;
    font-size:28px;
    font-weight:700;
    text-align:center;
    margin-top:10px;
}

.sidebar-menu{
    background:#2A245D;
    padding:15px;
    border-radius:15px;
    color:white;
    font-size:18px;
    margin-bottom:10px;
    font-weight:600;
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
    line-height:1.2;
}

.sub-title{
    color:#64748B;
    font-size:17px;
    margin-bottom:20px;
}

/* Filter title */
.filter-title{
    color:#0F172A !important;
    font-size:24px !important;
    font-weight:800 !important;
    margin-bottom:10px;
}

/* ===========================
KPI CARD
=========================== */

[data-testid="metric-container"]{
    background:#FFFFFF !important;
    border-radius:22px !important;
    padding:24px !important;
    border:1px solid #DCE3EC !important;
    box-shadow:0 6px 16px rgba(0,0,0,0.10) !important;
    min-height:130px !important;
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


/* Filter */
.stSelectbox label{
    color:#0F172A !important;
    font-weight:700 !important;
}

.stSelectbox div[data-baseweb="select"]{
    background:#FFFFFF !important;
    border-radius:12px !important;
    border:1px solid #CBD5E1 !important;
}

/* Plotly chart */
.js-plotly-plot{
    background:#FFFFFF !important;
    border-radius:18px !important;
    padding:8px !important;
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
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-title">
        AYAM SERAYU
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1,4])

    with c1:
        if st.button("🏠", key="dashboard_btn"):
            st.switch_page("pages/dashboard.py")

    with c2:
        st.markdown(
            "<div class='menu-text'>Dashboard</div>",
            unsafe_allow_html=True
        )

    c3, c4 = st.columns([1,4])

    with c3:
        if st.button("🔮", key="prediksi_btn"):
            st.switch_page("pages/prediksi.py")

    with c4:
        st.markdown(
            "<div class='menu-text'>Prediksi</div>",
            unsafe_allow_html=True
        )
# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class='main-title'>
DASHBOARD INTERAKTIF<br>
PENJUALAN PRODUK UMKM AYAM SERAYU
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub-title'>
Dataset ini merupakan data transaksi penjualan Ayam 
Serayu periode 2023 sampai 2025 yang terdiri dari 626.311 baris data, 206.296 transaksi unik, 3 outlet,
dan 17 produk dalam kategori makanan serta minuman. Dataset bersumber dari Kaggle dan tidak memiliki missing value, sehingga layak digunakan untuk analisis pola penjualan dan 
prediksi permintaan produk
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================================================
# FILTER LOGIC (HIDDEN)
# =========================================================

tahun = st.session_state.get("tahun", "Semua")
bulan = st.session_state.get("bulan", "Semua")
outlet = st.session_state.get("outlet", "Semua")
produk = st.session_state.get("produk", "Semua")

filtered_df = df.copy()

if tahun != "Semua":
    filtered_df = filtered_df[
        filtered_df["tahun"] == tahun
    ]

if bulan != "Semua":
    filtered_df = filtered_df[
        filtered_df["bulan"] == bulan
    ]

if outlet != "Semua":
    filtered_df = filtered_df[
        filtered_df["Outlet"] == outlet
    ]

if produk != "Semua":
    filtered_df = filtered_df[
        filtered_df["Nama Produk"] == produk
    ]
# =========================================================
# KPI
# =========================================================

total_dataset = len(filtered_df)

avg_penjualan = (
    filtered_df
    .groupby("Tanggal")["Jumlah Produk"]
    .sum()
    .mean()
)

if pd.isna(avg_penjualan):
    avg_penjualan = 0

produk_terlaris = (
    filtered_df.groupby("Nama Produk")
    ["Jumlah Produk"]
    .sum()
)

if len(produk_terlaris) > 0:
    produk_terlaris = produk_terlaris.idxmax()
else:
    produk_terlaris = "-"

jumlah_transaksi = len(filtered_df)

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Total Dataset",
    f"{total_dataset:,}"
)

c2.metric(
    "Rata-rata Penjualan Perhari",
    f"{avg_penjualan:.0f}"
)

c3.metric(
    "Produk Terlaris",
    produk_terlaris
)

c4.metric(
    "Jumlah Transaksi",
    f"{jumlah_transaksi:,}"
)

st.write("")
# =========================================================
# FILTER + CHART ATAS
# =========================================================

left_filter, left_chart, right_chart = st.columns([1,2,2])


# =====================
# FILTER KIRI
# =====================

with left_filter:

    st.markdown(
        "<div class='filter-title'>Filter Data</div>",
        unsafe_allow_html=True
    )

    tahun = st.selectbox(
        "Tahun",
        ["Semua"] + sorted(
            list(df["tahun"].unique())
        ),
        key="tahun"
    )

    bulan = st.selectbox(
        "Bulan",
        ["Semua"] + sorted(
            list(df["bulan"].unique())
        ),
        key="bulan"
    )

    outlet = st.selectbox(
        "Outlet",
        ["Semua"] + sorted(
            list(df["Outlet"].dropna().unique())
        ),
        key="outlet"
    )

    produk = st.selectbox(
        "Produk",
        ["Semua"] + sorted(
            list(df["Nama Produk"].dropna().unique())
        ),
        key="produk"
    )



# =====================
# CHART TREN HARIAN
# =====================

trend = (
    filtered_df
    .groupby("Tanggal")["Jumlah Produk"]
    .sum()
    .reset_index()
)

fig1 = px.line(
    trend,
    x="Tanggal",
    y="Jumlah Produk",
    title="Tren Penjualan Harian",
    template="simple_white"
)

fig1.update_layout(
    paper_bgcolor="#E4EFF9",
    plot_bgcolor="#E4EFF9",

    font=dict(
        color="black",
        size=14
    ),

    title_font=dict(
        color="black",
        size=18
    ),

    xaxis=dict(
        title="Tanggal",
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showgrid=True,
        gridcolor="#8499B9"
    ),

    yaxis=dict(
        title="Jumlah Produk",
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showgrid=True,
        gridcolor="#405E8B"
    )
)

fig1.update_traces(
    line_color="#1E2487",
    line_width=3
)

with left_chart:
    st.plotly_chart(
        fig1,
        use_container_width=True
    )

# =========================================================
# RATA-RATA PENJUALAN BULANAN PER TAHUN
# SAMA PERSIS DENGAN NOTEBOOK
# =========================================================

daily_sales = (
    filtered_df
    .groupby("Tanggal")["Jumlah Produk"]
    .sum()
    .reset_index()
)

daily_sales["tahun"] = daily_sales["Tanggal"].dt.year
daily_sales["bulan"] = daily_sales["Tanggal"].dt.month

avg_bulanan_tahun = (
    daily_sales
    .groupby(["tahun", "bulan"])["Jumlah Produk"]
    .mean()
    .reset_index()
)

avg_bulanan_tahun.columns = [
    "tahun",
    "bulan",
    "target"
]

fig2 = go.Figure()

colors = [
    "#A7C7E7",
    "#F7C6C7",
    "#B5EAD7"
]

for i, tahun in enumerate(
    sorted(avg_bulanan_tahun["tahun"].unique())
):

    temp = avg_bulanan_tahun[
        avg_bulanan_tahun["tahun"] == tahun
    ]

    fig2.add_trace(
        go.Scatter(
            x=temp["bulan"],
            y=temp["target"],
            mode="lines+markers",
            name=str(tahun),

            line=dict(
                width=3,
                color=colors[i]
            ),

            marker=dict(
                size=7
            )
        )
    )

fig2.update_layout(
    title="Rata-rata Penjualan Bulanan per Tahun",

    xaxis_title="Bulan",
    yaxis_title="Rata-rata Penjualan",

    template="plotly_white",
    hovermode="x unified",

    paper_bgcolor="#E4EFF9",
    plot_bgcolor="#E4EFF9",

    font=dict(
        color="black",
        size=14
    ),

    title_font=dict(
        color="black",
        size=18
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(
            color="black",
            size=12
        )
    ),

    xaxis=dict(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showgrid=True,
        gridcolor="#8499B9"
    ),

    yaxis=dict(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showgrid=True,
        gridcolor="#8499B9"
    )
)

fig2.update_xaxes(dtick=1)

with right_chart:
    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =========================================================
# CHART BAWAH
# =========================================================

b1,b2,b3 = st.columns(3)

# Produk Terlaris
produk_chart = (
    filtered_df.groupby("Nama Produk")
    ["Jumlah Produk"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

fig3 = px.pie(
    values=produk_chart.values,
    names=produk_chart.index,
    title="Produk Terlaris"
)
fig3.update_layout(
    paper_bgcolor="#E4EFF9",
    plot_bgcolor="#9DB0D6",

    font=dict(
        color="black",
        size=14
    ),

    title_font=dict(
        color="black",
        size=18
    ),

    legend=dict(
        font=dict(
            color="black",
            size=12
        )
    )
)

with b1:
    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# Per Tahun
year_chart = (
    filtered_df.groupby("tahun")
    ["Jumlah Produk"]
    .sum()
    .reset_index()
)

fig4 = px.bar(
    year_chart,
    x="tahun",
    y="Jumlah Produk",
    title="Perbandingan Penjualan per Tahun",
    template="simple_white"
)

fig4.update_layout(
    paper_bgcolor="#E4EFF9",
    plot_bgcolor="#E4EFF9",

    font=dict(
        color="black",
        size=14
    ),

    title_font=dict(
        color="black",
        size=18
    ),

    xaxis=dict(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showgrid=True,
        gridcolor="#8499B9"
    ),

    yaxis=dict(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showgrid=True,
        gridcolor="#405E8B"
    )
)
fig4.update_traces(
marker_color="#45973A"
)

with b2:
    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# =========================================================
# PENJUALAN PER OUTLET (LINE CHART)
# =========================================================

outlet_monthly = (
    filtered_df
    .groupby(["Outlet", "bulan"])["Jumlah Produk"]
    .sum()
    .reset_index()
)

fig5 = go.Figure()

colors = [
    "#125A4A",
    "#1E2487",
    "#E67E22"
]

for i, outlet in enumerate(
    outlet_monthly["Outlet"].unique()
):

    temp = outlet_monthly[
        outlet_monthly["Outlet"] == outlet
    ]

    fig5.add_trace(
        go.Scatter(
            x=temp["bulan"],
            y=temp["Jumlah Produk"],
            mode="lines+markers",
            name=outlet,

            line=dict(
                width=3,
                color=colors[i % len(colors)]
            ),

            marker=dict(size=7)
        )
    )

fig5.update_layout(
    title="Tren Penjualan per Outlet",

    xaxis_title="Bulan",
    yaxis_title="Jumlah Produk",

    template="plotly_white",

    paper_bgcolor="#E4EFF9",
    plot_bgcolor="#E4EFF9",

    font=dict(
        color="black",
        size=14
    ),

    title_font=dict(
        color="black",
        size=18
    ),

    legend=dict(
        font=dict(
            color="black",
            size=12
        )
    ),

    xaxis=dict(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showgrid=True,
        gridcolor="#8499B9"
    ),

    yaxis=dict(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showgrid=True,
        gridcolor="#8499B9"
    )
)
with b3:
    st.plotly_chart(
        fig5,
        use_container_width=True
    )