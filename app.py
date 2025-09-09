import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="UK Listed Fintech Dashboard", layout="wide")

TICKERS = {
    "Wise plc (WISE.L)": "WISE.L",
    "Funding Circle (FCH.L)": "FCH.L",
    "OSB Group (OSB.L)": "OSB.L",
    "Plus500 (PLUS.L)": "PLUS.L",
}

@st.cache_data(ttl=3600)
def fetch_statements(ticker: str):
    t = yf.Ticker(ticker)
    fin = t.financials    # income statement (annual)
    bal = t.balance_sheet  # balance sheet (annual)
    return fin, bal

def compute_kpis(fin: pd.DataFrame, bal: pd.DataFrame):
    fin = fin.T.sort_index() if isinstance(fin, pd.DataFrame) else pd.DataFrame()
    bal = bal.T.sort_index() if isinstance(bal, pd.DataFrame) else pd.DataFrame()

    df = pd.DataFrame(index=fin.index)
    df["revenue"] = fin.get("Total Revenue")
    df["net_income"] = fin.get("Net Income")
    df["gross_profit"] = fin.get("Gross Profit")
    df["operating_income"] = fin.get("Operating Income")
    df["total_equity"] = bal.get("Total Stockholder Equity")
    df["total_debt"] = bal.get("Total Debt")

    # Ratios
    df["net_margin_pct"] = (df["net_income"] / df["revenue"]) * 100
    df["roe_pct"] = (df["net_income"] / df["total_equity"]) * 100
    df["debt_to_equity"] = df["total_debt"] / df["total_equity"]
    df = df.dropna(how="all")
    return df

st.title("UK Listed Fintech Dashboard")

col1, col2 = st.columns([2,1])
with col1:
    pick = st.multiselect("Pick companies", list(TICKERS.keys()), default=list(TICKERS.keys()))
with col2:
    st.caption("Data: Yahoo Finance via yfinance (annual statements)")

all_frames = []
for name in pick:
    fin, bal = fetch_statements(TICKERS[name])
    df = compute_kpis(fin, bal)
    if not df.empty:
        df = df.reset_index(names="period_end")
        df["ticker"] = name
        all_frames.append(df)

if not all_frames:
    st.warning("No data returned from Yahoo Finance for the current selection.")
    st.stop()

data = pd.concat(all_frames, ignore_index=True)

# KPI tiles (latest period per ticker)
latest = data.sort_values("period_end").groupby("ticker").tail(1)
k1,k2,k3,k4,k5 = st.columns(5)
def nice_money(x): 
    return "£{:,.0f}M".format(x/1e6) if pd.notna(x) else "—"

k1.metric("Revenue (latest)", " | ".join(nice_money(x) for x in latest["revenue"]))
k2.metric("Net Income (latest)", " | ".join(nice_money(x) for x in latest["net_income"]))
k3.metric("Net Margin", " | ".join(f"{x:.1f}%" if pd.notna(x) else "—" for x in latest["net_margin_pct"]))
k4.metric("ROE", " | ".join(f"{x:.1f}%" if pd.notna(x) else "—" for x in latest["roe_pct"]))
k5.metric("Debt/Equity", " | ".join(f"{x:.2f}" if pd.notna(x) else "—" for x in latest["debt_to_equity"]))

st.divider()

# Profitability chart
st.subheader("Profitability Trend")
fig = px.bar(
    data.dropna(subset=["revenue"]),
    x="ticker", y="revenue", color="ticker",
    facet_col="period_end", facet_col_wrap=4,
    title="Revenue by Year",
)
st.plotly_chart(fig, use_container_width=True)

fig2 = px.bar(
    data.dropna(subset=["net_margin_pct"]),
    x="ticker", y="net_margin_pct", color="ticker",
    facet_col="period_end", facet_col_wrap=4,
    title="Net Margin % by Year"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Leverage Trend (Debt/Equity)")
fig3 = px.line(
    data.dropna(subset=["debt_to_equity"]),
    x="period_end", y="debt_to_equity", color="ticker", markers=True
)
st.plotly_chart(fig3, use_container_width=True)