"""
Enterprise Crypto Exposure Dashboard
------------------------------------

This Streamlit experience positions the data as a real business command center
for a digital asset treasury or crypto-focused fund. Executives can scan the
headline health of tracked assets, while quants and ops teams can dive into
market drivers, sentiment flow, and audit readiness in dedicated tabs.
"""

from datetime import timedelta
import sqlite3
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="Crypto Exposure Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DATABASE CONNECTION ---
conn = sqlite3.connect("crypto_analytics.db")


@st.cache_data
def load_data():
    """Load combined metrics from SQLite."""
    df = pd.read_sql("SELECT * FROM combined_metrics", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


@st.cache_data
def load_sentiment_data():
    """Load sentiment index history."""
    df = pd.read_sql("SELECT * FROM sentiment_index", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


# --- HELPERS -----------------------------------------------------------------
def pct_delta(series: pd.Series, periods: int = 7) -> Optional[float]:
    """Return percent change for the last value versus N periods back."""
    if series is None or len(series.dropna()) <= periods:
        return None
    clean = series.dropna()
    latest = clean.iloc[-1]
    baseline = clean.iloc[-periods - 1]
    if baseline == 0:
        return None
    return (latest / baseline - 1) * 100


def format_large_number(value: float, prefix: str = "") -> str:
    """Pretty-print large values for KPI tiles."""
    if value is None or pd.isna(value):
        return "—"
    abs_val = abs(value)
    if abs_val >= 1e12:
        formatted = f"{value / 1e12:.1f}T"
    elif abs_val >= 1e9:
        formatted = f"{value / 1e9:.1f}B"
    elif abs_val >= 1e6:
        formatted = f"{value / 1e6:.1f}M"
    elif abs_val >= 1e3:
        formatted = f"{value / 1e3:.1f}K"
    else:
        formatted = f"{value:.0f}"
    return f"{prefix}{formatted}"


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest row per symbol from a filtered DataFrame."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.sort_values("date")
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values("symbol")
    )


def window_return(grp: pd.DataFrame, days: int) -> Optional[float]:
    """Compute price return for the last N calendar days within a group."""
    if grp.empty or grp["Close"].isna().all():
        return None
    ordered = grp.sort_values("date")
    recent_cutoff = ordered["date"].max() - timedelta(days=days)
    window_df = ordered[ordered["date"] >= recent_cutoff]
    if window_df.empty:
        return None
    start_price = window_df["Close"].iloc[0]
    end_price = ordered["Close"].iloc[-1]
    if start_price == 0:
        return None
    return (end_price / start_price - 1) * 100


# --- LOAD DATA ----------------------------------------------------------------
data = load_data()
sentiment_data = load_sentiment_data()
coins = sorted(data["symbol"].unique())

# --- SIDEBAR CONTROLS ---------------------------------------------------------
st.sidebar.title("Control Tower")
default_assets = ["BTC-USD", "ETH-USD", "SOL-USD"]
tracked_assets = st.sidebar.multiselect(
    "Tracked assets",
    options=coins,
    default=[c for c in default_assets if c in coins][:3] or coins[:2],
)

window_options = {"Last 90 days": 90, "Last 180 days": 180, "Year to date": 365, "Full history": None}
lookback_label = st.sidebar.selectbox("Lookback window", list(window_options.keys()), index=0)
lookback_days = window_options[lookback_label]
performance_window_options = {"7D momentum": 7, "30D trend": 30, "90D view": 90}
performance_label = st.sidebar.select_slider(
    "Performance focus",
    options=list(performance_window_options.keys()),
    value="30D trend",
)
performance_window = performance_window_options[performance_label]

if tracked_assets:
    filtered = data[data["symbol"].isin(tracked_assets)].copy()
else:
    filtered = pd.DataFrame(columns=data.columns)

if not filtered.empty and lookback_days:
    min_allowed = filtered["date"].max() - timedelta(days=lookback_days)
    filtered = filtered[filtered["date"] >= min_allowed]

sentiment_filtered = sentiment_data
if lookback_days:
    min_sentiment_date = sentiment_filtered["date"].max() - timedelta(days=lookback_days)
    sentiment_filtered = sentiment_filtered[sentiment_filtered["date"] >= min_sentiment_date]

st.sidebar.markdown(
    f"**Assets in scope:** `{len(tracked_assets)}`\n\n"
    f"**Records displayed:** `{len(filtered):,}`"
)

# --- HEADER -------------------------------------------------------------------
st.title("📊 Digital Asset Exposure Command Center")
st.caption(
    "Live health, sentiment, and audit signals for treasury, trading, and risk stakeholders."
)

# Create business-aligned tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Briefing",
        "Market Drivers",
        "Sentiment & Flow",
        "Data Quality",
    ]
)

# --- TAB 1: EXECUTIVE BRIEFING ------------------------------------------------
with tab1:
    st.subheader("Portfolio Command Center")
    if filtered.empty:
        st.info("Select at least one asset on the left to populate the dashboard.")
    else:
        daily_health = filtered.groupby("date")["market_health"].mean().dropna()
        daily_volume = filtered.groupby("date")["Volume"].sum().dropna()
        daily_volatility = filtered.groupby("date")["volatility"].mean().dropna()
        latest_sentiment = sentiment_filtered["fear_greed"].iloc[-1] if not sentiment_filtered.empty else None
        sentiment_delta = pct_delta(sentiment_filtered["fear_greed"]) if not sentiment_filtered.empty else None

        col1, col2, col3, col4 = st.columns(4)
        if not daily_health.empty:
            health_value = daily_health.iloc[-1]
            health_delta_pct = pct_delta(daily_health, periods=performance_window // 7 if performance_window >= 7 else 1)
            col1.metric(
                "Portfolio Health (avg)",
                format_large_number(health_value, prefix="$"),
                None if health_delta_pct is None else f"{health_delta_pct:.1f}% vs prior window",
            )
        if not daily_volatility.empty:
            vol_value = daily_volatility.iloc[-1] * 100
            vol_delta = pct_delta(daily_volatility, periods=5)
            col2.metric(
                "Realized Volatility",
                f"{vol_value:.2f}%",
                None if vol_delta is None else f"{vol_delta:.1f}% change",
            )
        if not daily_volume.empty:
            volume_value = daily_volume.iloc[-1]
            volume_delta = pct_delta(daily_volume, periods=5)
            col3.metric(
                "Daily Notional Flow",
                format_large_number(volume_value, prefix="$"),
                None if volume_delta is None else f"{volume_delta:.1f}% WoW",
            )
        if latest_sentiment is not None:
            col4.metric(
                "Fear & Greed Index",
                f"{latest_sentiment:.0f}/100",
                None if sentiment_delta is None else f"{sentiment_delta:.1f}% vs last week",
            )

        st.markdown("##### Health Trajectory")
        if not daily_health.empty:
            fig_health = px.area(
                daily_health.reset_index(),
                x="date",
                y="market_health",
                title="Aggregated Market Health Score",
            )
            fig_health.update_traces(line_color="#0068c9")
            st.plotly_chart(fig_health, use_container_width=True)

        # Asset-level scorecard
        snapshot = latest_snapshot(filtered)
        if not snapshot.empty:
            # Compute rolling returns for spotlight
            performance_rows = []
            for symbol, grp in filtered.groupby("symbol"):
                ret = window_return(grp, performance_window)
                performance_rows.append(
                    {
                        "symbol": symbol,
                        "window_return": ret,
                        "latest_close": grp.sort_values("date")["Close"].iloc[-1],
                        "avg_volatility": grp["volatility"].tail(7).mean(),
                        "market_health": grp["market_health"].iloc[-1],
                        "Volume": grp["Volume"].iloc[-1],
                    }
                )
            perf_df = pd.DataFrame(performance_rows).dropna(subset=["window_return"])

            st.markdown("##### Opportunity & Risk Radar")
            if not perf_df.empty:
                movers = perf_df.sort_values("window_return", ascending=False)
                fig_movers = px.bar(
                    movers,
                    x="symbol",
                    y="window_return",
                    color="window_return",
                    color_continuous_scale="BrBG",
                    title=f"{performance_label} by asset",
                    labels={"window_return": "Return (%)"},
                )
                fig_movers.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_movers, use_container_width=True)

                st.markdown("##### Risk/Return Positioning")
                fig_scatter = px.scatter(
                    perf_df,
                    x="avg_volatility",
                    y="window_return",
                    size="Volume",
                    color="symbol",
                    hover_data=["market_health"],
                    labels={
                        "avg_volatility": "7D Volatility",
                        "window_return": "Window Return (%)",
                        "Volume": "Latest Volume",
                    },
                    title="Risk vs Reward (bubble size = daily volume)",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info(
                    "Not enough lookback history for the selected window to compute momentum. "
                    "Try expanding the lookback or picking a shorter performance focus."
                )

            st.markdown("##### Latest Position Snapshot")
            display_cols = ["symbol", "date", "Close", "returns", "volatility", "fear_greed", "market_health"]
            formatted_snapshot = snapshot[display_cols].copy()
            formatted_snapshot["returns"] = formatted_snapshot["returns"] * 100
            formatted_snapshot["volatility"] = formatted_snapshot["volatility"] * 100
            st.dataframe(
                formatted_snapshot.style.format(
                    {
                        "Close": "${:,.2f}",
                        "returns": "{:+.2f}%",
                        "volatility": "{:.2f}%",
                        "fear_greed": "{:.0f}",
                        "market_health": "${:,.0f}",
                    }
                )
            )

# --- TAB 2: MARKET DRIVERS ----------------------------------------------------
with tab2:
    st.subheader("Market Drivers & Diagnostics")
    if filtered.empty:
        st.info("Use the sidebar filters to review price action and drivers.")
    else:
        metric_map = {
            "Closing price (USD)": "Close",
            "Volume (USD)": "Volume",
            "Realized volatility": "volatility",
            "Market health score": "market_health",
        }
        selected_metric = st.selectbox("Driver to analyze", list(metric_map.keys()))
        metric_column = metric_map[selected_metric]

        fig_driver = px.line(
            filtered,
            x="date",
            y=metric_column,
            color="symbol",
            title=f"{selected_metric} trend",
        )
        if metric_column == "Volume":
            fig_driver.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_driver, use_container_width=True)

        st.markdown("##### Liquidity Pulse")
        liquidity = (
            filtered.groupby("date")["Volume"]
            .sum()
            .reset_index()
            .rename(columns={"Volume": "daily_volume"})
        )
        fig_liquidity = px.bar(
            liquidity,
            x="date",
            y="daily_volume",
            title="Aggregate traded notional",
        )
        fig_liquidity.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_liquidity, use_container_width=True)

        st.markdown("##### Cross-Metric Correlations")
        numerical_cols = ["Close", "Volume", "volatility", "returns", "volume_change", "fear_greed", "market_health"]
        corr_data = filtered[numerical_cols].corr()
        fig_corr_matrix = px.imshow(
            corr_data,
            labels=dict(x="Metric", y="Metric", color="Correlation"),
            x=numerical_cols,
            y=numerical_cols,
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
        )
        st.plotly_chart(fig_corr_matrix, use_container_width=True)

# --- TAB 3: SENTIMENT & FLOW --------------------------------------------------
with tab3:
    st.subheader("Sentiment & Flow Intelligence")
    if sentiment_filtered.empty:
        st.warning("No sentiment feed available. Run the ETL job to refresh data.")
    else:
        fig_sentiment = px.line(
            sentiment_filtered,
            x="date",
            y="fear_greed",
            title="Fear & Greed Index trend",
            labels={"fear_greed": "Score (0-100)"}
        )
        fig_sentiment.update_layout(yaxis=dict(range=[0, 100]))
        fig_sentiment.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.08, annotation_text="Extreme fear")
        fig_sentiment.add_hrect(y0=25, y1=46, fillcolor="orange", opacity=0.08, annotation_text="Fear")
        fig_sentiment.add_hrect(y0=46, y1=54, fillcolor="yellow", opacity=0.08, annotation_text="Neutral")
        fig_sentiment.add_hrect(y0=54, y1=75, fillcolor="lightgreen", opacity=0.08, annotation_text="Greed")
        fig_sentiment.add_hrect(y0=75, y1=100, fillcolor="green", opacity=0.08, annotation_text="Extreme greed")
        st.plotly_chart(fig_sentiment, use_container_width=True)

        latest_sentiment = sentiment_filtered["fear_greed"].iloc[-1]
        avg_sentiment = sentiment_filtered["fear_greed"].mean()
        sentiment_7d = pct_delta(sentiment_filtered["fear_greed"], periods=7)

        col1, col2, col3 = st.columns(3)
        col1.metric("Current sentiment", f"{latest_sentiment:.0f}/100")
        col2.metric("Long-run average", f"{avg_sentiment:.1f}/100")
        if sentiment_7d is not None:
            col3.metric("7D sentiment change", f"{sentiment_7d:.1f}%")

        if latest_sentiment < 25:
            st.error("Desk sentiment: **Extreme fear** — liquidity providers may demand wider spreads.")
        elif latest_sentiment < 46:
            st.warning("Desk sentiment: **Risk-off** — maintain tighter position limits.")
        elif latest_sentiment > 75:
            st.success("Desk sentiment: **Extreme greed** — consider harvesting gains.")
        elif latest_sentiment > 54:
            st.info("Desk sentiment: **Risk-on** — flows skew bullish.")
        else:
            st.info("Desk sentiment: **Neutral** — balanced positioning.")

        if filtered.empty:
            st.info("Add at least one asset to relate sentiment to price action.")
        else:
            st.markdown("##### Sentiment vs Price Response")
            merged_sentiment = pd.merge(
                filtered,
                sentiment_filtered,
                on="date",
                how="left",
                suffixes=("", "_sentiment"),
            )
            fig_dual = go.Figure()
            for symbol in tracked_assets:
                symbol_data = merged_sentiment[merged_sentiment["symbol"] == symbol]
                if symbol_data.empty:
                    continue
                fig_dual.add_trace(
                    go.Scatter(
                        x=symbol_data["date"],
                        y=symbol_data["Close"],
                        mode="lines",
                        name=f"{symbol} price",
                    )
                )
            fig_dual.add_trace(
                go.Scatter(
                    x=sentiment_filtered["date"],
                    y=sentiment_filtered["fear_greed"],
                    mode="lines",
                    name="Fear & Greed",
                    yaxis="y2",
                    line=dict(dash="dot", color="gray"),
                )
            )
            fig_dual.update_layout(
                title="Price vs sentiment overlay",
                yaxis=dict(title="Price (USD)"),
                yaxis2=dict(
                    title="Sentiment score",
                    overlaying="y",
                    side="right",
                    range=[0, 100],
                ),
            )
            st.plotly_chart(fig_dual, use_container_width=True)

            st.markdown("##### Flow responsiveness")
            corr_value = merged_sentiment["Close"].corr(merged_sentiment["fear_greed"])
            st.write(f"Correlation between sentiment and closing price: `{corr_value:.3f}`")
            fig_corr = px.scatter(
                merged_sentiment,
                x="fear_greed",
                y="Close",
                color="symbol",
                title="Sentiment vs price",
                labels={"fear_greed": "Fear & Greed", "Close": "Price"},
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 4: DATA QUALITY ------------------------------------------------------
with tab4:
    st.subheader("Data Quality & Audit Trail")
    try:
        audit = pd.read_sql(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 20",
            conn,
        )
        if audit.empty:
            st.info("Audit table is empty. Run `audit_checks.py` after the ETL refresh.")
        else:
            issues_open = audit["issues_found"].sum()
            st.metric("Outstanding issues (last 20 checks)", issues_open)
            st.dataframe(audit)
    except Exception:
        st.warning("Audit log unavailable. Ensure `audit_log` table exists.")

# Close connection
conn.close()