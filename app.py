"""
app.py  –  Tour Package Sales EDA Dashboard
Run:  streamlit run app.py
"""
from __future__ import annotations
import pathlib, sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.data_loader import CAT_COLS, NUM_COLS, TARGET, COLUMN_DESCRIPTIONS, load_clean
from src.analysis import (
    conversion_summary, conversion_by, numeric_summary,
    product_performance, income_by_designation, age_band_conversion,
    passport_owncar_conversion, correlation_with_target,
    followup_income_pivot, mean_by, distribution,
)

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH = pathlib.Path(__file__).parent / "data" / "tour_package.csv"

# Dark-mode palette — vivid enough to read on #0F172A
C_BLUE  = "#60A5FA"   # blue-400
C_ORG   = "#FB923C"   # orange-400
C_GREEN = "#4ADE80"   # green-400
C_PURP  = "#A78BFA"   # violet-400
C_TEAL  = "#22D3EE"   # cyan-400
PALETTE = [C_BLUE, C_ORG, C_GREEN, C_PURP, C_TEAL, "#FACC15", "#F472B6"]

# Dark-mode chart constants
DK_BG   = "#0F172A"   # main bg
DK_SURF = "#1E293B"   # surface / card bg
DK_GRID = "#334155"   # grid lines
DK_TEXT = "#E2E8F0"   # primary text
DK_MUTE = "#94A3B8"   # muted text

st.set_page_config(
    page_title="Tour Package Sales · EDA Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (dark mode) ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Main areas ── */
[data-testid="stAppViewContainer"] { background: #0F172A; }
[data-testid="stMain"]             { background: #0F172A; }
[data-testid="block-container"]    { background: #0F172A; }

/* ── Sidebar ── */
[data-testid="stSidebar"]          { background: #1E293B !important; }
[data-testid="stSidebar"] *        { color: #E2E8F0 !important; }
[data-testid="stSidebar"] label    { color: #94A3B8 !important; font-size:12px; }

/* ── Streamlit native text ── */
html, body, [class*="css"]         { color: #E2E8F0; }
p, li, span, label                 { color: #E2E8F0; }

/* ── KPI cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 16px;
}
.kpi-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,.4);
}
.kpi-card .icon  { font-size:24px; margin-bottom:6px; }
.kpi-card .label { font-size:11px; font-weight:600; letter-spacing:.06em;
                   text-transform:uppercase; color:#64748B; margin-bottom:4px; }
.kpi-card .val   { font-size:32px; font-weight:800; color:#E2E8F0; line-height:1; }
.kpi-card .sub   { font-size:11px; color:#64748B; margin-top:4px; }
.kpi-card.green  .val { color:#4ADE80; }
.kpi-card.orange .val { color:#FB923C; }
.kpi-card.blue   .val { color:#60A5FA; }
.kpi-card.purple .val { color:#A78BFA; }

/* ── Section headers ── */
.sec-header {
    font-size:17px; font-weight:700; color:#E2E8F0;
    border-left:4px solid #60A5FA; padding-left:10px;
    margin: 20px 0 10px;
}

/* ── Insight boxes ── */
.insight-box {
    background:#1E3A5F; border:1px solid #1D4ED8;
    border-radius:10px; padding:14px 18px; margin-top:10px;
}
.insight-box p { margin:4px 0; font-size:13px; color:#BAE6FD; }
.insight-box p span { font-weight:700; color:#7DD3FC; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-size:13px !important; font-weight:600 !important;
    padding:8px 18px !important;
    color: #94A3B8 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #60A5FA !important;
    border-bottom-color: #60A5FA !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] { background: #1E293B; }
</style>
""", unsafe_allow_html=True)


# ── Data ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning data...")
def get_data(p):
    return load_clean(p)

df_full = get_data(DATA_PATH)


# ── Sidebar filters ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Tour Package EDA")
    st.markdown("<hr style='border-color:#334155'>", unsafe_allow_html=True)
    st.markdown("### Filters")

    sel_gender = st.multiselect(
        "Gender", sorted(df_full["Gender"].unique()),
        default=sorted(df_full["Gender"].unique()))

    sel_occ = st.multiselect(
        "Occupation", sorted(df_full["Occupation"].unique()),
        default=sorted(df_full["Occupation"].unique()))

    sel_city = st.multiselect(
        "City Tier", sorted(df_full["CityTier"].unique()),
        default=sorted(df_full["CityTier"].unique()))

    sel_desg = st.multiselect(
        "Designation", sorted(df_full["Designation"].unique()),
        default=sorted(df_full["Designation"].unique()))

    # MaritalStatus — 2nd strongest chi-square predictor (χ²=185.6, p<0.001)
    sel_marital = st.multiselect(
        "Marital Status", sorted(df_full["MaritalStatus"].unique()),
        default=sorted(df_full["MaritalStatus"].unique()))

    # Passport — doubles conversion rate (passport holders: ~29% vs ~17%)
    sel_passport = st.multiselect(
        "Passport", ["No", "Yes"], default=["No", "Yes"])

    imin = int(df_full["MonthlyIncome"].min())
    imax = int(df_full["MonthlyIncome"].max())
    sel_income = st.slider("Monthly Income (INR)", imin, imax, (imin, imax), step=500)

    st.markdown("<hr style='border-color:#334155'>", unsafe_allow_html=True)
    st.caption(f"Dataset: **tour_package.csv**\nCleaned rows: **{len(df_full):,}**")


# ── Apply filters ──────────────────────────────────────────────────────────
_passport_vals = [0 if v == "No" else 1 for v in sel_passport]
df = df_full[
    df_full["Gender"].isin(sel_gender)
    & df_full["Occupation"].isin(sel_occ)
    & df_full["CityTier"].isin(sel_city)
    & df_full["Designation"].isin(sel_desg)
    & df_full["MaritalStatus"].isin(sel_marital)
    & df_full["Passport"].isin(_passport_vals)
    & df_full["MonthlyIncome"].between(sel_income[0], sel_income[1])
].copy()

if df.empty:
    st.warning("No data matches the current filters. Adjust the sidebar selections.")
    st.stop()


# ── Chart layout helper (dark mode) ───────────────────────────────────────
def chart_layout(fig, height=340, margin=None):
    m = margin or dict(t=40, b=30, l=10, r=10)
    fig.update_layout(
        height=height, margin=m,
        paper_bgcolor=DK_BG, plot_bgcolor=DK_SURF,
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=DK_TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=DK_TEXT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=DK_GRID, tickcolor=DK_GRID,
                     tickfont=dict(color=DK_MUTE), title_font=dict(color=DK_MUTE))
    fig.update_yaxes(gridcolor=DK_GRID, linecolor=DK_GRID,
                     tickfont=dict(color=DK_MUTE), title_font=dict(color=DK_MUTE))
    return fig


# ── Page header ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1E3A5F 0%,#2563EB 100%);
     border-radius:14px;padding:28px 32px;margin-bottom:20px;'>
  <h1 style='color:#fff;margin:0;font-size:28px;font-weight:800;letter-spacing:-.5px'>
    ✈️ Tour Package Sales — EDA Dashboard
  </h1>
  <p style='color:#BAE6FD;margin:6px 0 0;font-size:14px'>
    Exploratory analysis of customer purchase behaviour · {len(df_full):,} records · 20 features
  </p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "📊 Overview",
    "👤 Demographics",
    "🎯 Product & Pitch",
    "💰 Income & Designation",
    "🔗 Correlations",
    "📐 Statistical Tests",
    "🔎 Data Explorer",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with t1:
    kpi = conversion_summary(df)

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card blue">
        <div class="icon">👥</div>
        <div class="label">Total Customers</div>
        <div class="val">{kpi['total_customers']:,}</div>
        <div class="sub">Filtered dataset</div>
      </div>
      <div class="kpi-card green">
        <div class="icon">✅</div>
        <div class="label">Packages Sold</div>
        <div class="val">{kpi['packages_sold']:,}</div>
        <div class="sub">ProdTaken = 1</div>
      </div>
      <div class="kpi-card orange">
        <div class="icon">❌</div>
        <div class="label">Not Sold</div>
        <div class="val">{kpi['not_sold']:,}</div>
        <div class="sub">ProdTaken = 0</div>
      </div>
      <div class="kpi-card purple">
        <div class="icon">📈</div>
        <div class="label">Conversion Rate</div>
        <div class="val">{kpi['conversion_rate']}%</div>
        <div class="sub">Sold ÷ Total</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sec-header'>Purchase Split</div>", unsafe_allow_html=True)
        vc = df[TARGET].value_counts().reset_index()
        vc.columns = ["Status", "Count"]
        vc["Status"] = vc["Status"].map({0: "Not Taken", 1: "Taken"})
        fig = px.pie(vc, values="Count", names="Status", hole=0.55,
                     color="Status",
                     color_discrete_map={"Taken": C_GREEN, "Not Taken": C_BLUE})
        fig.update_traces(textinfo="percent+label", textfont_size=13,
                          marker=dict(line=dict(color=DK_BG, width=2)))
        fig.add_annotation(
            text=f"<b>{kpi['conversion_rate']}%</b><br><span style='font-size:11px'>Conversion</span>",
            x=0.5, y=0.5, showarrow=False, font_size=18, align="center")
        chart_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='sec-header'>Conversion by City Tier</div>", unsafe_allow_html=True)
        ct = conversion_by(df, "CityTier").sort_values("CityTier")
        ct["CityTier"] = "Tier " + ct["CityTier"].astype(str)
        fig = go.Figure()
        fig.add_bar(x=ct["CityTier"], y=ct["Total"], name="Total",
                    marker_color="#1E3A5F", text=ct["Total"], textposition="inside")
        fig.add_bar(x=ct["CityTier"], y=ct["Sold"], name="Sold",
                    marker_color=C_BLUE, text=ct["Sold"], textposition="inside")
        for _, row in ct.iterrows():
            fig.add_annotation(x=row["CityTier"], y=row["Total"] + 40,
                               text=f"<b>{row['ConversionRate']}%</b>",
                               showarrow=False, font=dict(color=C_ORG, size=12))
        fig.update_layout(barmode="overlay", bargap=0.35,
                          legend=dict(orientation="h", y=1.12))
        chart_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='sec-header'>Conversion by Occupation</div>", unsafe_allow_html=True)
        occ = conversion_by(df, "Occupation")
        fig = px.bar(occ, x="Occupation", y="ConversionRate",
                     text="ConversionRate", color="ConversionRate",
                     color_continuous_scale=[DK_SURF, C_BLUE])
        fig.update_traces(texttemplate="%{text}%", textposition="outside",
                          marker_line_color=DK_BG, marker_line_width=1)
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-15)
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        # Warn about Free Lancer outlier (n=2 in full dataset → 100% conversion)
        _fl = occ[occ["Occupation"] == "Free Lancer"]
        if not _fl.empty and _fl.iloc[0]["Total"] <= 5:
            st.caption(
                f"⚠️ **Free Lancer** shows {_fl.iloc[0]['ConversionRate']}% conversion "
                f"but has only {int(_fl.iloc[0]['Total'])} customer(s) — treat as statistical noise."
            )

    with col4:
        st.markdown("<div class='sec-header'>Conversion by Marital Status</div>", unsafe_allow_html=True)
        ms = conversion_by(df, "MaritalStatus")
        fig = px.bar(ms, x="MaritalStatus", y="ConversionRate",
                     text="ConversionRate", color="MaritalStatus",
                     color_discrete_sequence=PALETTE)
        fig.update_traces(texttemplate="%{text}%", textposition="outside",
                          marker_line_color=DK_BG, marker_line_width=1)
        fig.update_layout(showlegend=False)
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Insight box
    _ct_raw   = conversion_by(df, "CityTier").sort_values("ConversionRate", ascending=False)
    ct_best   = _ct_raw.iloc[0]
    prod_best = product_performance(df).sort_values("ConversionRate", ascending=False).iloc[0]
    age_best  = age_band_conversion(df).sort_values("ConversionRate", ascending=False).iloc[0]
    occ_best  = conversion_by(df, "Occupation").sort_values("ConversionRate", ascending=False).iloc[0]
    ms_best   = conversion_by(df, "MaritalStatus").sort_values("ConversionRate", ascending=False).iloc[0]

    st.markdown(f"""
    <div class="insight-box">
      <p>💡 <span>Overall conversion rate</span> is <span>{kpi['conversion_rate']}%</span>
         — only 1 in 5 customers purchases a package.</p>
      <p>🏙️ <span>City Tier {ct_best['CityTier']}</span> leads conversions at
         <span>{ct_best['ConversionRate']}%</span>.</p>
      <p>🎯 <span>{prod_best['ProductPitched']}</span> package has the highest conversion
         at <span>{prod_best['ConversionRate']}%</span>.</p>
      <p>👶 Age band <span>{age_best['AgeBand']}</span> converts best at
         <span>{age_best['ConversionRate']}%</span> — youngest cohort drives the most sales.</p>
      <p>💼 <span>{occ_best['Occupation']}</span> customers have the highest purchase rate
         at <span>{occ_best['ConversionRate']}%</span>.</p>
      <p>💍 <span>{ms_best['MaritalStatus']}</span> customers convert most at
         <span>{ms_best['ConversionRate']}%</span> (chi-square p &lt; 0.001).</p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEMOGRAPHICS
# ════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("<div class='sec-header'>Age Band Conversion</div>", unsafe_allow_html=True)
    age_df = age_band_conversion(df)
    fig = px.bar(age_df, x="AgeBand", y="ConversionRate",
                 text="ConversionRate", color="ConversionRate",
                 color_continuous_scale=[DK_SURF, C_BLUE],
                 custom_data=["Total", "Sold"])
    fig.update_traces(
        texttemplate="%{text}%", textposition="outside",
        hovertemplate="<b>%{x}</b><br>Conversion: %{y}%<br>Total: %{customdata[0]}<br>Sold: %{customdata[1]}<extra></extra>",
        marker_line_color=DK_BG, marker_line_width=1.5,
    )
    fig.update_layout(coloraxis_showscale=False)
    chart_layout(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sec-header'>Gender Conversion</div>", unsafe_allow_html=True)
        gen = conversion_by(df, "Gender")
        fig = go.Figure()
        fig.add_bar(name="Total", x=gen["Gender"], y=gen["Total"],
                    marker_color="#1E3A5F", text=gen["Total"], textposition="inside")
        fig.add_bar(name="Sold", x=gen["Gender"], y=gen["Sold"],
                    marker_color=C_BLUE, text=gen["Sold"], textposition="inside")
        for _, row in gen.iterrows():
            fig.add_annotation(x=row["Gender"], y=row["Total"] + 30,
                               text=f"<b>{row['ConversionRate']}%</b>",
                               showarrow=False, font=dict(color=C_ORG, size=12))
        fig.update_layout(barmode="overlay", bargap=0.4,
                          legend=dict(orientation="h", y=1.1))
        chart_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='sec-header'>Passport & Own Car Effect</div>", unsafe_allow_html=True)
        pc = passport_owncar_conversion(df)
        fig = px.bar(pc, x="Feature", y="ConversionRate", color="Value",
                     barmode="group", text="ConversionRate",
                     color_discrete_map={"No": "#93C5FD", "Yes": C_ORG},
                     custom_data=["Total"])
        fig.update_traces(
            texttemplate="%{text}%", textposition="outside",
            hovertemplate="<b>%{x} - %{fullData.name}</b><br>Conversion: %{y}%<br>Customers: %{customdata[0]}<extra></extra>",
            marker_line_color=DK_BG, marker_line_width=1,
        )
        chart_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='sec-header'>Age Distribution by Outcome</div>", unsafe_allow_html=True)
    df_plot = df.copy()
    df_plot["Outcome"] = df_plot[TARGET].map({0: "Not Taken", 1: "Taken"})
    fig = px.histogram(df_plot, x="Age", color="Outcome", nbins=35,
                       barmode="overlay", opacity=0.75,
                       color_discrete_map={"Not Taken": C_BLUE, "Taken": C_ORG},
                       marginal="box")
    chart_layout(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)

    # Age-band insight from analysis Section 12
    _age_conv = age_band_conversion(df).sort_values("ConversionRate", ascending=False)
    _age_top  = _age_conv.iloc[0]
    _mean_taken    = round(df[df[TARGET] == 1]["Age"].mean(), 1)
    _mean_nottaken = round(df[df[TARGET] == 0]["Age"].mean(), 1)
    _pass_yes = round(df[df["Passport"] == 1][TARGET].mean() * 100, 1)
    _pass_no  = round(df[df["Passport"] == 0][TARGET].mean() * 100, 1)
    st.markdown(f"""
    <div class="insight-box">
      <p>👶 Age band <span>{_age_top['AgeBand']}</span> has the highest conversion at
         <span>{_age_top['ConversionRate']}%</span>. Mean age: Taken={_mean_taken} vs Not Taken={_mean_nottaken}
         (t-test p &lt; 0.001).</p>
      <p>🛂 Passport holders convert at <span>{_pass_yes}%</span> vs <span>{_pass_no}%</span> without a passport
         — nearly <span>{round(_pass_yes/_pass_no, 1)}×</span> higher rate.</p>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='sec-header'>Occupation Split</div>", unsafe_allow_html=True)
        occ_d = distribution(df, "Occupation")
        fig = px.pie(occ_d, values="Count", names="Occupation", hole=0.45,
                     color_discrete_sequence=PALETTE)
        fig.update_traces(textinfo="percent+label",
                          marker=dict(line=dict(color=DK_BG, width=2)))
        chart_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("<div class='sec-header'>Number of Trips by Outcome</div>", unsafe_allow_html=True)
        fig = px.box(df_plot, x="Outcome", y="NumberOfTrips",
                     color="Outcome", points="outliers",
                     color_discrete_map={"Not Taken": C_BLUE, "Taken": C_ORG})
        fig.update_layout(showlegend=False)
        chart_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRODUCT & PITCH
# ════════════════════════════════════════════════════════════════════════════
with t3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sec-header'>Product Pitched — Conversion Rate</div>", unsafe_allow_html=True)
        prod = product_performance(df)
        bar_colors = [C_GREEN if v == prod["ConversionRate"].max() else C_BLUE
                      for v in prod["ConversionRate"]]
        fig = go.Figure(go.Bar(
            x=prod["ProductPitched"], y=prod["ConversionRate"],
            text=[f"{v}%" for v in prod["ConversionRate"]],
            textposition="outside",
            marker_color=bar_colors,
            marker_line_color=DK_BG, marker_line_width=1.5,
            customdata=prod[["Total", "Sold"]].values,
            hovertemplate="<b>%{x}</b><br>Conversion: %{y}%<br>Total: %{customdata[0]}<br>Sold: %{customdata[1]}<extra></extra>",
        ))
        chart_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='sec-header'>Contact Type — Conversion</div>", unsafe_allow_html=True)
        ct2 = conversion_by(df, "TypeofContact")
        fig = px.bar(ct2, x="TypeofContact", y="ConversionRate",
                     text="ConversionRate", color="TypeofContact",
                     color_discrete_sequence=[C_TEAL, C_PURP],
                     custom_data=["Total", "Sold"])
        fig.update_traces(
            texttemplate="%{text}%", textposition="outside",
            hovertemplate="<b>%{x}</b><br>Conversion: %{y}%<br>Total: %{customdata[0]}<br>Sold: %{customdata[1]}<extra></extra>",
            marker_line_color=DK_BG, marker_line_width=1,
        )
        fig.update_layout(showlegend=False)
        chart_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='sec-header'>Follow-ups vs Conversion Rate & Mean Income</div>", unsafe_allow_html=True)
    fu = followup_income_pivot(df)
    fig = go.Figure()
    fig.add_bar(x=fu["NumberOfFollowups"], y=fu["ConversionRate"],
                name="Conversion %", marker_color=C_BLUE,
                text=fu["ConversionRate"], texttemplate="%{text}%",
                textposition="outside", yaxis="y1")
    fig.add_scatter(x=fu["NumberOfFollowups"], y=fu["MeanIncome"],
                    name="Mean Income (INR)", mode="lines+markers",
                    line=dict(color=C_ORG, width=2.5),
                    marker=dict(size=8), yaxis="y2")
    fig.update_layout(
        yaxis=dict(title="Conversion Rate (%)", showgrid=False),
        yaxis2=dict(title="Mean Income (INR)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.12), bargap=0.4,
    )
    chart_layout(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='sec-header'>Pitch Satisfaction vs Conversion</div>", unsafe_allow_html=True)
        ps = conversion_by(df, "PitchSatisfactionScore").sort_values("PitchSatisfactionScore")
        fig = go.Figure()
        fig.add_bar(x=ps["PitchSatisfactionScore"].astype(str), y=ps["Total"],
                    name="Total Customers", marker_color="#1E3A5F")
        fig.add_scatter(x=ps["PitchSatisfactionScore"].astype(str),
                        y=ps["ConversionRate"], name="Conversion %",
                        mode="lines+markers+text",
                        text=[f"{v}%" for v in ps["ConversionRate"]],
                        textposition="top center",
                        line=dict(color=C_ORG, width=2.5),
                        marker=dict(size=9, color=C_ORG), yaxis="y2")
        fig.update_layout(
            yaxis=dict(title="Customers", showgrid=False),
            yaxis2=dict(title="Conversion %", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12), bargap=0.4,
        )
        chart_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("<div class='sec-header'>Pitch Duration by Outcome</div>", unsafe_allow_html=True)
        df_plot2 = df.copy()
        df_plot2["Outcome"] = df_plot2[TARGET].map({0: "Not Taken", 1: "Taken"})
        fig = px.violin(df_plot2, x="Outcome", y="DurationOfPitch",
                        color="Outcome", box=True, points=False,
                        color_discrete_map={"Not Taken": C_BLUE, "Taken": C_ORG})
        fig.update_layout(showlegend=False)
        chart_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='sec-header'>Preferred Property Star — Conversion</div>", unsafe_allow_html=True)
    star = conversion_by(df, "PreferredPropertyStar").sort_values("PreferredPropertyStar")
    star["Stars"] = star["PreferredPropertyStar"].astype(str) + " star"
    fig = px.bar(star, x="Stars", y="ConversionRate",
                 text="ConversionRate", color="ConversionRate",
                 color_continuous_scale=[DK_SURF, C_PURP])
    fig.update_traces(texttemplate="%{text}%", textposition="outside",
                      marker_line_color=DK_BG, marker_line_width=1)
    fig.update_layout(coloraxis_showscale=False)
    chart_layout(fig, height=280)
    st.plotly_chart(fig, use_container_width=True)

    # Product & Pitch insight box from analysis Sections 4, 11, 14
    _fu_top    = followup_income_pivot(df).sort_values("ConversionRate", ascending=False).iloc[0]
    _prod_best = product_performance(df).sort_values("ConversionRate", ascending=False).iloc[0]
    _dur_taken    = round(df[df[TARGET] == 1]["DurationOfPitch"].mean(), 1)
    _dur_nottaken = round(df[df[TARGET] == 0]["DurationOfPitch"].mean(), 1)
    st.markdown(f"""
    <div class="insight-box">
      <p>📞 <span>{int(_fu_top['NumberOfFollowups'])} follow-up(s)</span> yields the highest conversion at
         <span>{_fu_top['ConversionRate']}%</span>. More follow-ups = higher close rate (significant, p &lt; 0.001).</p>
      <p>🎯 <span>{_prod_best['ProductPitched']}</span> is the top-converting product at
         <span>{_prod_best['ConversionRate']}%</span> — nearly 4× the rate of Super Deluxe.</p>
      <p>⏱️ Longer pitches correlate with sales: Taken avg <span>{_dur_taken} min</span>
         vs Not Taken <span>{_dur_nottaken} min</span> (t-test p &lt; 0.001).</p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — INCOME & DESIGNATION
# ════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("<div class='sec-header'>Median Monthly Income by Designation</div>", unsafe_allow_html=True)
    inc = income_by_designation(df)
    fig = go.Figure(go.Bar(
        x=inc["Designation"], y=inc["MedianIncome"],
        text=[f"INR {v:,.0f}" for v in inc["MedianIncome"]],
        textposition="outside",
        marker=dict(
            color=inc["MedianIncome"],
            colorscale=[[0, DK_SURF], [1, C_BLUE]],
            line=dict(color=DK_GRID, width=1.5),
        ),
    ))
    chart_layout(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sec-header'>Designation — Conversion Rate</div>", unsafe_allow_html=True)
        dconv = conversion_by(df, "Designation")
        fig = px.bar(dconv, x="Designation", y="ConversionRate",
                     text="ConversionRate", color="Designation",
                     color_discrete_sequence=PALETTE)
        fig.update_traces(texttemplate="%{text}%", textposition="outside",
                          marker_line_color=DK_BG, marker_line_width=1)
        fig.update_layout(showlegend=False, xaxis_tickangle=-20)
        chart_layout(fig, height=330)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='sec-header'>Income Distribution by Outcome</div>", unsafe_allow_html=True)
        df_inc = df.copy()
        df_inc["Outcome"] = df_inc[TARGET].map({0: "Not Taken", 1: "Taken"})
        fig = px.box(df_inc, x="Outcome", y="MonthlyIncome",
                     color="Outcome", notched=True, points=False,
                     color_discrete_map={"Not Taken": C_BLUE, "Taken": C_ORG})
        fig.update_layout(showlegend=False)
        chart_layout(fig, height=330)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='sec-header'>Mean Income by Occupation</div>", unsafe_allow_html=True)
        occ_inc = mean_by(df, "Occupation", "MonthlyIncome")
        fig = px.bar(occ_inc, x="Occupation", y="Mean_MonthlyIncome",
                     text="Mean_MonthlyIncome", color="Occupation",
                     color_discrete_sequence=PALETTE)
        fig.update_traces(texttemplate="INR %{text:,.0f}", textposition="outside",
                          marker_line_color=DK_BG, marker_line_width=1)
        fig.update_layout(showlegend=False)
        chart_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("<div class='sec-header'>Income vs Number of Trips</div>", unsafe_allow_html=True)
        df_sc = df.copy()
        df_sc["Outcome"] = df_sc[TARGET].map({0: "Not Taken", 1: "Taken"})
        sample = df_sc.sample(min(len(df_sc), 800), random_state=42)
        fig = px.scatter(sample, x="MonthlyIncome", y="NumberOfTrips",
                         color="Outcome", opacity=0.65,
                         color_discrete_map={"Not Taken": C_BLUE, "Taken": C_ORG},
                         labels={"MonthlyIncome": "Monthly Income (INR)",
                                 "NumberOfTrips": "No. of Trips/Year"})
        chart_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — CORRELATIONS
# ════════════════════════════════════════════════════════════════════════════
with t5:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sec-header'>Feature Correlation with Target (Signed)</div>", unsafe_allow_html=True)
        # Build signed correlation series so direction is visible
        _corr_raw = df[NUM_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
        corr_signed = pd.DataFrame({
            "Feature":     _corr_raw.index,
            "Correlation": _corr_raw.round(4).values,
            "Direction":   ["Negative" if v < 0 else "Positive" for v in _corr_raw.values],
        })
        fig = px.bar(corr_signed, x="Correlation", y="Feature",
                     orientation="h", text="Correlation",
                     color="Direction",
                     color_discrete_map={"Negative": C_ORG, "Positive": C_BLUE})
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                          marker_line_color=DK_BG, marker_line_width=1)
        fig.add_vline(x=0, line_width=1, line_color="#94A3B8")
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          legend_title_text="Direction")
        chart_layout(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='sec-header'>Correlation Matrix (Heatmap)</div>", unsafe_allow_html=True)
        corr_m = df[NUM_COLS + [TARGET]].corr().round(2)
        fig = px.imshow(corr_m, text_auto=".2f",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto")
        fig.update_layout(
            height=380, margin=dict(t=30, b=10, l=10, r=10),
            paper_bgcolor=DK_BG, plot_bgcolor=DK_SURF,
            font=dict(color=DK_TEXT),
            coloraxis_colorbar=dict(thickness=12, tickfont=dict(color=DK_TEXT)),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='sec-header'>3D Scatter: Age × Income × Trips (by Outcome)</div>",
                unsafe_allow_html=True)
    df_3d = df.copy()
    df_3d["Outcome"] = df_3d[TARGET].map({0: "Not Taken", 1: "Taken"})
    sample_3d = df_3d.sample(min(len(df_3d), 1000), random_state=1)
    fig = px.scatter_3d(sample_3d, x="Age", y="MonthlyIncome", z="NumberOfTrips",
                        color="Outcome", opacity=0.65,
                        color_discrete_map={"Not Taken": C_BLUE, "Taken": C_ORG},
                        labels={"MonthlyIncome": "Income (INR)"})
    fig.update_layout(height=480, margin=dict(t=30, b=10, l=0, r=0),
                      scene=dict(
                          xaxis=dict(backgroundcolor=DK_SURF),
                          yaxis=dict(backgroundcolor=DK_SURF),
                          zaxis=dict(backgroundcolor=DK_SURF),
                      ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='sec-header'>Sales Funnel — Customers by Follow-up Count</div>",
                unsafe_allow_html=True)
    fu2 = followup_income_pivot(df).sort_values("NumberOfFollowups")
    fig = go.Figure(go.Funnel(
        y=[f"{int(r)} follow-up(s)" for r in fu2["NumberOfFollowups"]],
        x=fu2["Customers"],
        textinfo="value+percent initial",
        marker=dict(color=[C_BLUE, C_TEAL, C_GREEN, C_PURP, C_ORG, "#CA8A04"][:len(fu2)]),
    ))
    chart_layout(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — STATISTICAL TESTS
# ════════════════════════════════════════════════════════════════════════════
with t6:
    from scipy import stats as _stats

    st.markdown("<div class='sec-header'>Chi-Square Tests — Categorical Features vs Purchase Decision</div>",
                unsafe_allow_html=True)
    st.caption("Tests whether each categorical feature is statistically independent of ProdTaken.")

    chi_rows = []
    for col in CAT_COLS:
        ct_chi = pd.crosstab(df[col], df[TARGET])
        chi2, p, dof, _ = _stats.chi2_contingency(ct_chi)
        chi_rows.append({
            "Feature":     col,
            "Chi²":        round(chi2, 2),
            "p-value":     round(p, 6),
            "df":          dof,
            "Significant": "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else "—",
            "Verdict":     "Linked to purchase ✓" if p < 0.05 else "No significant link",
        })
    chi_df = pd.DataFrame(chi_rows).sort_values("p-value")

    # Colour p-values: significant rows highlighted
    def _style_chi(row):
        color = "#D1FAE5" if row["p-value"] < 0.05 else ""
        return [f"background-color:{color}"] * len(row)

    st.dataframe(
        chi_df.style.apply(_style_chi, axis=1).format({"Chi²": "{:.2f}", "p-value": "{:.6f}"}),
        hide_index=True, use_container_width=True, height=260,
    )

    # Visualise chi-square statistic
    fig_chi = px.bar(chi_df.sort_values("Chi²"), x="Chi²", y="Feature",
                     orientation="h", text="Chi²",
                     color="Chi²", color_continuous_scale=[DK_SURF, C_PURP])
    fig_chi.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                          marker_line_color=DK_BG, marker_line_width=1)
    fig_chi.update_layout(coloraxis_showscale=False,
                          yaxis={"categoryorder": "total ascending"})
    chart_layout(fig_chi, height=300)
    st.plotly_chart(fig_chi, use_container_width=True)

    st.markdown("<div class='sec-header'>Independent T-Tests — Numeric Features vs Purchase Decision</div>",
                unsafe_allow_html=True)
    st.caption("Tests whether the mean of each numeric feature differs significantly between Taken and Not Taken groups.")

    _taken     = df[df[TARGET] == 1]
    _not_taken = df[df[TARGET] == 0]
    ttest_rows = []
    for col in NUM_COLS:
        t_stat, p_val = _stats.ttest_ind(_taken[col], _not_taken[col], equal_var=False)
        ttest_rows.append({
            "Feature":        col,
            "Mean (Taken)":   round(_taken[col].mean(), 3),
            "Mean (Not)":     round(_not_taken[col].mean(), 3),
            "Difference":     round(_taken[col].mean() - _not_taken[col].mean(), 3),
            "T-Statistic":    round(t_stat, 3),
            "p-value":        round(p_val, 6),
            "Significant":    "★★★" if p_val < 0.001 else "★★" if p_val < 0.01 else "★" if p_val < 0.05 else "—",
        })
    ttest_df = pd.DataFrame(ttest_rows).sort_values("p-value")

    def _style_tt(row):
        color = "#D1FAE5" if row["p-value"] < 0.05 else ""
        return [f"background-color:{color}"] * len(row)

    st.dataframe(
        ttest_df.style.apply(_style_tt, axis=1).format({
            "Mean (Taken)": "{:.3f}", "Mean (Not)": "{:.3f}",
            "Difference": "{:.3f}", "T-Statistic": "{:.3f}", "p-value": "{:.6f}",
        }),
        hide_index=True, use_container_width=True, height=340,
    )

    # Visualise absolute T-statistic
    ttest_df["Abs T"] = ttest_df["T-Statistic"].abs()
    fig_tt = px.bar(ttest_df.sort_values("Abs T"), x="Abs T", y="Feature",
                    orientation="h", text="T-Statistic",
                    color="Abs T", color_continuous_scale=[DK_SURF, C_TEAL])
    fig_tt.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                         marker_line_color=DK_BG, marker_line_width=1)
    fig_tt.update_layout(coloraxis_showscale=False,
                         yaxis={"categoryorder": "total ascending"})
    fig_tt.add_vline(x=1.96, line_dash="dash", line_color="#94A3B8",
                     annotation_text="t=1.96 (p≈0.05)", annotation_position="top right")
    chart_layout(fig_tt, height=320)
    st.plotly_chart(fig_tt, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
      <p>📊 All 6 categorical features are statistically linked to purchase (chi-square p &lt; 0.05).</p>
      <p>📈 6 of 9 numeric features differ significantly between Taken/Not Taken groups (t-test p &lt; 0.05):
         <span>Age, DurationOfPitch, NumberOfFollowups, PreferredPropertyStar, MonthlyIncome, PitchSatisfactionScore</span>.</p>
      <p>❌ <span>NumberOfTrips, NumberOfPersonVisiting, NumberOfChildrenVisiting</span> show no significant difference.</p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with t7:
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("<div class='sec-header'>Column Dictionary</div>", unsafe_allow_html=True)
        desc = pd.DataFrame(list(COLUMN_DESCRIPTIONS.items()), columns=["Column", "Description"])
        st.dataframe(desc, height=400, hide_index=True, use_container_width=True)

    with col_r:
        st.markdown("<div class='sec-header'>Numeric Summary Statistics</div>", unsafe_allow_html=True)
        st.dataframe(numeric_summary(df, NUM_COLS), height=400, use_container_width=True)

    st.markdown("<div class='sec-header'>Dataset Preview</div>", unsafe_allow_html=True)
    n = st.slider("Rows to show", 10, 300, 25, step=5)
    st.dataframe(df.head(n), use_container_width=True, hide_index=True)

    st.markdown("<div class='sec-header'>Category Explorer</div>", unsafe_allow_html=True)
    sel_cat = st.selectbox("Select a categorical column", CAT_COLS)
    dist_df = distribution(df, sel_cat)
    ca, cb = st.columns(2)
    with ca:
        st.dataframe(dist_df, hide_index=True, width="stretch")
    with cb:
        fig = px.bar(dist_df, x=sel_cat, y="Count", text="Pct",
                     color=sel_cat, color_discrete_sequence=PALETTE)
        fig.update_traces(texttemplate="%{text}%", textposition="outside",
                          marker_line_color=DK_BG, marker_line_width=1)
        fig.update_layout(showlegend=False)
        chart_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Download button
    st.markdown("---")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="tour_package_filtered.csv",
        mime="text/csv",
    )
