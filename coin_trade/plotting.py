from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _fmt_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _trade_window(df: pd.DataFrame, trades: pd.DataFrame, margin_bars: int) -> pd.DataFrame:
    if trades.empty:
        raise ValueError("Trades DataFrame is empty")

    start = max(0, int(trades["entry_idx"].min()) - margin_bars)
    end = min(len(df) - 1, int(trades["exit_idx"].max()) + margin_bars)
    return df.iloc[start : end + 1].copy()


def render_trades_plot(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    title: str = "Backtest",
    output_path: str | None = None,
    margin_bars: int = 25,
    trade_index: int | None = None,
) -> go.Figure:
    """Render a Plotly figure highlighting entries, TP, and SL zones."""
    if trades.empty:
        raise ValueError("Trades DataFrame is empty.")

    selected_trades = trades.iloc[[trade_index]] if trade_index is not None else trades
    df_window = _trade_window(df, selected_trades, margin_bars)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_window.index,
                open=df_window["open"],
                high=df_window["high"],
                low=df_window["low"],
                close=df_window["close"],
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                showlegend=False,
            )
        ]
    )

    shapes = []
    annotations = []
    scatter_x = []
    scatter_y = []
    scatter_color = []
    scatter_text = []

    color_map = {"tp": "#2ecc71", "sl": "#e74c3c", "open": "#f1c40f"}

    for _, tr in selected_trades.iterrows():
        entry_t = tr["entry_time"]
        exit_t = tr["exit_time"]
        entry = float(tr["entry_price"])
        tp = float(tr["tp_price"])
        sl = float(tr["sl_price"])
        exit_price = float(tr["exit_price"])
        outcome = tr["outcome"]
        gross = float(tr.get("gross_return_pct", np.nan))
        net = float(tr.get("net_return_pct", np.nan))
        tp_pct = float(tr.get("tp_pct", np.nan))
        sl_pct = float(tr.get("sl_pct", np.nan))

        shapes.extend(
            [
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=entry_t,
                    x1=exit_t,
                    y0=entry,
                    y1=tp,
                    line=dict(color="rgba(46,204,113,0.7)", width=1),
                    fillcolor="rgba(46,204,113,0.18)",
                    layer="below",
                ),
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=entry_t,
                    x1=exit_t,
                    y0=sl,
                    y1=entry,
                    line=dict(color="rgba(231,76,60,0.7)", width=1),
                    fillcolor="rgba(231,76,60,0.18)",
                    layer="below",
                ),
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=entry_t,
                    x1=exit_t,
                    y0=entry,
                    y1=entry,
                    line=dict(color="rgba(160,160,160,0.6)", dash="dash"),
                ),
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=entry_t,
                    x1=exit_t,
                    y0=tp,
                    y1=tp,
                    line=dict(color="rgba(46,204,113,0.8)", dash="dot"),
                ),
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=entry_t,
                    x1=exit_t,
                    y0=sl,
                    y1=sl,
                    line=dict(color="rgba(231,76,60,0.8)", dash="dot"),
                ),
            ]
        )

        annotations.extend(
            [
                dict(
                    x=entry_t,
                    y=entry,
                    xref="x",
                    yref="y",
                    text=f"<b>Entry</b><br>{entry:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowwidth=1,
                    ax=-45,
                    ay=-25,
                    bgcolor="rgba(220,220,220,0.92)",
                    bordercolor="rgba(80,80,80,0.6)",
                    borderwidth=1,
                    opacity=0.95,
                ),
                dict(
                    x=exit_t,
                    y=tp,
                    xref="x",
                    yref="y",
                    text=f"<b>TP</b><br>{tp:.2f} ({_fmt_pct(tp_pct)})",
                    showarrow=True,
                    arrowhead=2,
                    arrowwidth=1,
                    ax=-40,
                    ay=-22,
                    bgcolor="rgba(160,238,180,0.98)",
                    bordercolor="rgba(46,204,113,0.8)",
                    borderwidth=1,
                    opacity=0.98,
                ),
                dict(
                    x=exit_t,
                    y=sl,
                    xref="x",
                    yref="y",
                    text=f"<b>SL</b><br>{sl:.2f} ({_fmt_pct(sl_pct)})",
                    showarrow=True,
                    arrowhead=2,
                    arrowwidth=1,
                    ax=-40,
                    ay=22,
                    bgcolor="rgba(255,179,179,0.98)",
                    bordercolor="rgba(231,76,60,0.8)",
                    borderwidth=1,
                    opacity=0.98,
                ),
            ]
        )

        scatter_x.append(exit_t)
        scatter_y.append(exit_price)
        scatter_color.append(color_map.get(outcome, "#95a5a6"))
        scatter_text.append(
            f"{outcome.upper()}<br>Exit: {exit_price:.2f}<br>Gross: {_fmt_pct(gross)}<br>Net: {_fmt_pct(net)}"
        )

    fig.add_trace(
        go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode="markers",
            marker=dict(size=10, symbol="x", color=scatter_color, line=dict(width=1, color="#111")),
            text=scatter_text,
            hovertemplate="%{text}",
            name="Exit",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=True, rangeslider=dict(visible=True)),
        yaxis=dict(title="Price", showgrid=True),
        shapes=shapes,
        annotations=annotations,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig
