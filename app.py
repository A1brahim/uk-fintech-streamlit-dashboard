# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import math


# Minimal global styling
px.defaults.template = "simple_white"
px.defaults.color_discrete_sequence = [
    "#2E86AB", "#F6AA1C", "#E15554", "#3BB273"] 

st.set_page_config(page_title="UK Listed Fintech Dashboard", layout="wide")

# FinTech tickers (Yahoo Finance symbols)
TICKERS = {
    "Wise plc (WISE.L)": "WISE.L",
    "OSB Group (OSB.L)": "OSB.L",
    "Plus500 (PLUS.L)": "PLUS.L",
    "Funding Circle (FCH.L)": "FCH.L",
}

# Data helpers
@st.cache_data(ttl=24 * 3600, show_spinner=True)
def fetch_metrics(yf_symbol: str) -> pd.DataFrame:
    """
    Pull annual financials & balance sheet from yfinance and return tidy metrics by year.
    Returns empty DataFrame if data is missing.
    """
    t = yf.Ticker(yf_symbol)

    # Annual statements (columns are period end dates)
    fin = t.financials  # Total Revenue, Net Income
    bs = t.balance_sheet  # Total Debt, Total Stockholder Equity

    if fin is None or fin.empty or bs is None or bs.empty:
        return pd.DataFrame()

    # We want year as an integer; yfinance columns are Datetime-like
    fin = fin.copy()
    bs = bs.copy()

    # Rows we need; some tickers might use slightly different labels – try both common ones
    def first_present(df, candidates):
        for c in candidates:
            if c in df.index:
                return df.loc[c]
        return pd.Series(dtype="float64")

    revenue = first_present(fin, ["Total Revenue", "TotalRevenue"])
    net_income = first_present(fin, ["Net Income", "NetIncome"])
    total_debt = first_present(bs, ["Total Debt", "TotalDebt", "Total Liabilities"])  # debt can be missing
    equity = first_present(bs, ["Total Stockholder Equity", "Total Equity", "Total stockholders' equity"])

    # Build a long frame by year
    # Use the intersection of the available columns to align years
    cols = sorted(set(revenue.index).intersection(net_income.index))
    if len(cols) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Year": [pd.to_datetime(c).year for c in cols],
        "Revenue": [pd.to_numeric(revenue[c], errors="coerce") for c in cols],
        "Net Income": [pd.to_numeric(net_income[c], errors="coerce") for c in cols],
    })

    # Bring in debt/equity if available for the same columns/years
    if not total_debt.empty and not equity.empty:
        de_cols = sorted(set(total_debt.index).intersection(equity.index))
        if de_cols:
            de_map = {pd.to_datetime(c).year: (pd.to_numeric(total_debt[c], errors="coerce"),
                                              pd.to_numeric(equity[c], errors="coerce"))
                      for c in de_cols}
            df["Debt"] = df["Year"].map(lambda y: de_map.get(y, (pd.NA, pd.NA))[0])
            df["Equity"] = df["Year"].map(lambda y: de_map.get(y, (pd.NA, pd.NA))[1])
        else:
            df["Debt"] = pd.NA
            df["Equity"] = pd.NA
    else:
        df["Debt"] = pd.NA
        df["Equity"] = pd.NA

    # Derived metrics
    df["Net Margin %"] = (df["Net Income"] / df["Revenue"]) * 100
    df["Debt to Equity"] = pd.to_numeric(df["Debt"], errors="coerce") / pd.to_numeric(df["Equity"], errors="coerce")
    df["Ticker"] = yf_symbol
    return df.dropna(subset=["Revenue", "Net Income"], how="all")


@st.cache_data(ttl=24 * 3600, show_spinner=True)
def build_universe(selection: list[str]) -> pd.DataFrame:
    frames = []
    for label in selection:
        sym = TICKERS[label]
        d = fetch_metrics(sym)
        if d.empty:
            continue
        d = d.assign(FinTech=label)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values(["Year", "FinTech"])
    # Keep only sensible years (>= 2000)
    out = out[out["Year"] >= 2000]
    return out


# Sidebar filters 
st.sidebar.header("Filters")
selected_labels = st.sidebar.multiselect(
    "FinTech companies", list(TICKERS.keys()), default=list(TICKERS.keys())
)

data = build_universe(selected_labels)

if data.empty:
    st.warning("No financial data available from Yahoo Finance for the current selection.")
    st.stop()

min_year, max_year = int(data["Year"].min()), int(data["Year"].max())
year_range = st.sidebar.slider("Year range", min_year, max_year, (max(min_year, max_year-4), max_year))
view = data[(data["Year"] >= year_range[0]) & (data["Year"] <= year_range[1])].copy()

# KPI tiles (latest year in view) 
latest_year = int(view["Year"].max())
latest = view[view["Year"] == latest_year].copy()
st.markdown("### Listed UK FinTech KPI Tiles")
cols = st.columns([2, 2, 2, 2, 2])
cols[0].markdown(f"**Latest Year:** {latest_year}")

def fmt_currency(x): 
    try:
        # convert to billions for compactness
        return f"£{x/1e9:.1f}B"
    except Exception:
        return "—"

def fmt_pct(x):
    try:
        return f"{x:.0f}%"
    except Exception:
        return "—"

# Show one row per company
tile_table = latest[["FinTech", "Revenue", "Net Income", "Net Margin %", "Debt to Equity"]].copy()
tile_table["Revenue"] = tile_table["Revenue"].map(fmt_currency)
tile_table["Net Income"] = tile_table["Net Income"].map(fmt_currency)
tile_table["Net Margin %"] = tile_table["Net Margin %"].map(fmt_pct)
tile_table["Debt to Equity"] = tile_table["Debt to Equity"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
st.dataframe(tile_table, width="stretch", hide_index=True)

st.divider()

# Charts 
def style_minimal(fig, ytitle=None):
    fig.update_layout(
        template="simple_white",
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, Arial, sans-serif", size=12),

        # Give the title room and keep the plot tidy
        title=dict(
            y=0.98,
            x=0,
            xanchor="left",
            yanchor="top",
            pad=dict(t=4, b=12, l=0, r=0)
        ),

        margin=dict(l=10, r=10, t=90, b=20), 
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
        
    fig.update_yaxes(
        title=ytitle if ytitle else None,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        showline=False,
        ticks="outside",
        showticklabels=True,
        rangemode="tozero",)

    fig.update_xaxes(showgrid=False, zeroline=False)
    return fig

# Define y-axis ceiling helper
def nice_ceiling(value: float, step: float) -> float:
    """Round value up to the next multiple of `step`."""
    if value is None or pd.isna(value):
        return step
    return math.ceil(value / step) * step

def set_shared_facet_yaxis(fig, series: pd.Series,
                           scale: float = 1.0,
                           step: float = None,
                           min_floor: float = 0.0,
                           ticksuffix: str = "",
                           tickformat: str | None = None):
    """
    Share y across all facets and apply a clean range & ticks.

    - `series`: the raw data column (e.g., view["Revenue"])
    - `scale`: divide values by this (e.g., 1e6 to show millions)
    - `step`: tick spacing in *display* units (after scaling). If None, auto-pick.
    - `min_floor`: min y (usually 0)
    - `ticksuffix`: e.g., "m", "B", "%"
    - `tickformat`: e.g., ",.0f" for integers with thousands separators
    """
    # max in display units
    vmax_disp = float(series.max() / scale) if len(series) else 0.0

    # choose a step if not provided (aim ~5–6 ticks using 1/2/5*10^n)
    if step is None:
        target_ticks = 6
        raw = vmax_disp / max(1, target_ticks)
        mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
        base = raw / mag
        step = (1 if base <= 1 else 2 if base <= 2 else 5) * mag

    ymax_disp = max(step, nice_ceiling(vmax_disp, step))

    fig.update_yaxes(
        matches="y",
        range=[min_floor, ymax_disp],
        tick0=min_floor,
        dtick=step,
        ticksuffix=ticksuffix,
        tickformat=tickformat,   # e.g., ",.0f"
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        showticklabels=True,
        title_standoff=10,
    )

# Category order for consistent ordering in legends and facets
order=list(TICKERS.keys())

# REVENUE (by FinTech and Year) 
rev_fig = px.bar(
    view,
    x="FinTech",
    y=view["Revenue"]/1e6,  # in millions for readability
    color="FinTech",
    facet_col="Year",
    facet_col_wrap=2,
    facet_col_spacing=0.15,
    facet_row_spacing=0.08,
    title="Revenue (by FinTech and Year)",
    category_orders={"FinTech": order},
)

# Shared Y axis with clean ticks
set_shared_facet_yaxis(
    rev_fig, series=view["Revenue"], scale=1e6, step=500, tickformat=",.0f"
)

rev_fig.update_layout(
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="right",
        x=5,
        itemwidth=90,
        tracegroupgap=4,
    )
)

# format Y axis as £m (label says £m; ticks display in millions)
rev_fig.update_yaxes(
    title="Revenue (£m)",
    tickmode="array",
    matches="y",
    showticklabels=True,
    showgrid=True,
    gridcolor="rgba(0,0,0,0.06)",
    zeroline=False,
    title_standoff=10,   
)

# set category order after the figure is created
order = list(TICKERS.keys())
rev_fig.update_xaxes(categoryorder="array", categoryarray=order)


# cleaner bars/hover
rev_fig.update_traces(
    marker_line_width=0,
    hovertemplate="£%{y:,.0f}m<extra>%{x}</extra>"
)



rev_fig.update_xaxes(showgrid=False, ticks="")


# legend safely BELOW the entire figure, left aligned
rev_fig.update_layout(
    # generous margins so title/legend never collide
    margin=dict(l=10, r=10, t=70, b=120),
    bargap=0.25,
    height=700,
    title=dict(
        y=0.98, x=0, xanchor="left", yanchor="top",
        pad=dict(t=4, b=6, l=0, r=0)),
)

# prettier facet titles: "2023" instead of "Year=2023"
# and nudge them a bit downward so they’re not hugging the top line
rev_fig.for_each_annotation(lambda a: a.update(
    text=a.text.split("=")[-1],
    font=dict(size=14),
    y=a.y - 0.04
))

st.plotly_chart(rev_fig, use_container_width=True, config={"displayModeBar": False})

# Net Margin %
nm_fig = px.bar(
    view,
    x="FinTech",
    y="Net Margin %",
    color="FinTech",
    facet_col="Year",
    facet_col_wrap=2,
    facet_col_spacing=0.15,
    facet_row_spacing=0.08,          
    title="Profitability (Net Margin %)",
    category_orders={"FinTech": order}, )

nm_fig.update_layout(
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="right",
        x=5,
        itemwidth=90,
        tracegroupgap=4,
    )
)

# set category order after the figure is created
nm_fig.update_xaxes(categoryorder="array", categoryarray=order)

nm_fig.update_traces(
    marker_line_width=0,
    hovertemplate="%{y:.0f}%<extra>%{x}</extra>"
)

nm_fig.update_yaxes(
    title="Net Margin (%)",
    tickformat=".0f",
    ticksuffix="%"
)

nm_fig.update_layout(
    margin=dict(l=10, r=10, t=70, b=120),
    bargap=0.25,
    height=700,
    title=dict(
        y=0.98, x=0, xanchor="left", yanchor="top",
        pad=dict(t=4, b=6, l=0, r=0)),
)  

nm_fig.add_hline(
    y=0,                                
    line_color="red",                   
    line_dash="dot",                   
    line_width=2.0                        
)

nm_fig.update_xaxes(showgrid=False, ticks="")
nm_fig.update_yaxes(showgrid=True, 
                    gridcolor="rgba(0,0,0,0.06)", 
                    matches="y",
                    showticklabels=True,
                    zeroline=False)

nm_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=14)))

st.plotly_chart(nm_fig, use_container_width=True, config={"displayModeBar": False})

# Leverage (Debt to Equity) – show a line over years per company

# consistent category order
ORDER = list(TICKERS.keys()) 

# show the raw data rows we have for Debt/Equity
st.markdown("### Debt to Equity (Leverage) Data Points")
lines_df = view.dropna(subset=["Debt to Equity"]).sort_values(["FinTech", "Year"])

if lines_df.empty:
    st.info("No Debt/Equity data available for the current filters.")

else:
    lev_fig = px.line(
        lines_df,
        x="Year",
        y="Debt to Equity",
        color="FinTech",
        markers=True,
        title="Leverage Trend (Debt to Equity)",
        category_orders={"FinTech": ORDER},
    )

    lev_fig.update_traces(
        mode="lines+markers",
        line=dict(width=2),
        hovertemplate="Year=%{x}<br>Debt/Equity=%{y:.2f}<extra>%{fullData.name}</extra>"
    )

# Year ticks as integers (no 2,023.5, etc.)
    lev_fig.update_xaxes(tickmode="linear", dtick=1, tickformat=",d")

# Clean, shared y-axis range & ticks (works fine even without facets)
    
    set_shared_facet_yaxis(
        lev_fig, 
        series=lines_df["Debt to Equity"], 
        scale=1.0, 
        step=0.5, 
        tickformat=".2f")


# legend safely BELOW the entire figure, left aligned
    lev_fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            traceorder="normal"
        ),
        margin=dict(l=10, r=10, t=70, b=40),
    )

 # Light dotted horizontal guides (safe even if tickvals isn’t precomputed)
    tickvals = lev_fig.layout.yaxis.tickvals
    if tickvals:
        for y in tickvals:
            lev_fig.add_shape(
                type="line", xref="paper", x0=0, x1=1,
                yref="y", y0=y, y1=y,
                line=dict(color="rgba(0,0,0,0.25)", width=1, dash="dot"),
                layer="below"
            )
    lev_fig.add_hline(y=0, line_color="rgba(0,0,0,0.35)", line_dash="dot", line_width=1)

# Minimal styling function
    style_minimal(lev_fig, "Debt / Equity")
    st.plotly_chart(lev_fig, use_container_width=True, config={"displayModeBar": False})


# Footer 
st.caption(
    "Data source: Yahoo Finance via yfinance. Some statements may be missing for certain tickers/years; "
    "metrics shown reflect available data only."
)
