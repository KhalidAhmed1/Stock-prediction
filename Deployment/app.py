import os
import sys
import traceback
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Fix paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "Experiments"))
sys.path.insert(0, os.path.dirname(__file__))

from model_loader import STOCKS, load_model_for_stock, load_data_for_stock, load_all_models
from predictor import predict_n_days_ahead, get_validation_actuals_and_preds

app = Dash(__name__, title="Stock LSTM Predictor", suppress_callback_exceptions=True)

print("Loading all models...")
load_all_models()
print("All models loaded.")

dashboard_cache = {}

# ─── Color palette ────────────────────────────────────────────────────────────
C = {
    "bg":       "#f5f7fa",
    "surface":  "#ffffff",
    "card":     "#f8f9fa",
    "border":   "#e2e8f0",
    "accent":   "#0066cc",
    "accent2":  "#00c853",
    "gold":     "#ff9800",
    "red":      "#d32f2f",
    "text":     "#1a202c",
    "muted":    "#4a5568",
    "purple":   "#6c5ce7",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(0,0,0,0.01)",
    font=dict(color=C["text"], family="'Inter', sans-serif", size=16),
    margin=dict(l=10, r=10, t=50, b=10),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.05)",
        linecolor=C["border"],
        tickfont=dict(color=C["muted"], size=15),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.05)",
        linecolor=C["border"],
        tickfont=dict(color=C["muted"], size=15),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=C["border"],
        borderwidth=1,
        font=dict(size=15, color=C["text"]),
    ),
    hovermode="x unified",
)


def styled_fig(**kwargs):
    fig = go.Figure(**kwargs)
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def get_stock_data(ticker, period="1y"):
    cache_key = f"{ticker}_{period}"
    if cache_key in dashboard_cache:
        return dashboard_cache[cache_key]
    try:
        hist = yf.Ticker(ticker).history(period=period)
        hist = hist.reset_index()
        hist.dropna(inplace=True)
        hist['Returns'] = hist['Close'].pct_change()
        hist['SMA20']   = hist['Close'].rolling(20).mean()
        hist['SMA50']   = hist['Close'].rolling(50).mean()
        hist['STD20']   = hist['Close'].rolling(20).std()
        hist['Upper']   = hist['SMA20'] + 2 * hist['STD20']
        hist['Lower']   = hist['SMA20'] - 2 * hist['STD20']
        hist['Volatility'] = hist['Returns'].rolling(20).std() * np.sqrt(252)
        hist['CumReturn']  = (1 + hist['Returns']).cumprod() - 1
        hist['DrawdownPeak'] = hist['Close'].cummax()
        hist['Drawdown']     = (hist['Close'] - hist['DrawdownPeak']) / hist['DrawdownPeak']
        dashboard_cache[cache_key] = hist
        return hist
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


# ─── Shared style helpers ─────────────────────────────────────────────────────
NAV_STYLE = {
    "background": C["surface"],
    "borderBottom": f"3px solid {C['accent']}",
    "padding": "0 50px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
    "height": "70px",
    "position": "sticky",
    "top": "0",
    "zIndex": "100",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
}

CARD = {
    "background": C["surface"],
    "border": f"2px solid {C['border']}",
    "borderRadius": "12px",
    "padding": "32px",
    "marginBottom": "25px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
}

TAB_STYLE = {
    "background": "transparent",
    "border": "none",
    "borderBottom": f"3px solid transparent",
    "color": C["muted"],
    "fontFamily": "'Inter', sans-serif",
    "fontSize": "16px",
    "fontWeight": "700",
    "letterSpacing": "0.5px",
    "padding": "18px 28px",
    "cursor": "pointer",
    "textTransform": "uppercase",
}

TAB_SELECTED = {
    **TAB_STYLE,
    "borderBottom": f"3px solid {C['accent']}",
    "color": C["text"],
    "background": "transparent",
}

DD_STYLE = {
    "fontFamily": "'Inter', sans-serif",
    "fontSize": "16px",
    "background": C["surface"],
    "border": f"2px solid {C['border']}",
    "borderRadius": "8px",
    "color": C["text"],
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "12px 16px",
    "fontFamily": "'Inter', sans-serif",
    "fontSize": "16px",
    "background": C["surface"],
    "border": f"2px solid {C['border']}",
    "borderRadius": "8px",
    "color": C["text"],
    "outline": "none",
}

BTN_PRIMARY = {
    "width": "100%",
    "padding": "13px 22px",
    "background": C["accent"],
    "color": "#fff",
    "border": "none",
    "borderRadius": "8px",
    "cursor": "pointer",
    "fontFamily": "'Inter', sans-serif",
    "fontSize": "16px",
    "fontWeight": "700",
    "letterSpacing": "0.5px",
    "transition": "all 0.2s",
}

BTN_SUCCESS = {
    **BTN_PRIMARY,
    "background": C["accent2"],
}

LABEL = {
    "display": "block",
    "fontSize": "14px",
    "fontWeight": "700",
    "letterSpacing": "0.5px",
    "textTransform": "uppercase",
    "color": C["text"],
    "marginBottom": "10px",
    "fontFamily": "'Inter', sans-serif",
}


def section_title(text):
    return html.Div(
        text,
        style={
            "fontSize": "18px",
            "fontWeight": "700",
            "letterSpacing": "1px",
            "textTransform": "uppercase",
            "color": C["accent"],
            "marginBottom": "20px",
            "fontFamily": "'Inter', sans-serif",
            "borderLeft": f"4px solid {C['accent']}",
            "paddingLeft": "14px",
        }
    )


# ─── Layout ───────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={
        "fontFamily": "'Inter', sans-serif",
        "background": C["bg"],
        "color": C["text"],
        "minHeight": "100vh",
    },
    children=[
        # Google Font
        html.Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
        ),

        # Navbar
        html.Div(style=NAV_STYLE, children=[
            html.Div(children=[
                html.Span("◈ ", style={"color": C["accent"], "fontSize": "24px", "fontWeight": "800"}),
                html.Span("LSTM", style={"color": C["text"], "fontWeight": "800", "fontSize": "24px", "letterSpacing": "2px"}),
                html.Span("PREDICT", style={"color": C["accent"], "fontWeight": "800", "fontSize": "24px", "letterSpacing": "2px"}),
            ]),
            html.Div(children=[
                html.Span("● ", style={"color": C["accent2"], "fontSize": "12px"}),
                html.Span("MODELS ONLINE", style={"color": C["muted"], "fontSize": "14px", "letterSpacing": "1px", "fontWeight": "600"}),
            ]),
        ]),

        # Main content
        html.Div(
            style={"maxWidth": "1400px", "margin": "0 auto", "padding": "40px 32px"},
            children=[
                dcc.Tabs(
                    id="main-tabs",
                    value="tab-forecast",
                    children=[
                        dcc.Tab(label="▸ FORECASTING", value="tab-forecast",
                                style=TAB_STYLE, selected_style=TAB_SELECTED),
                        dcc.Tab(label="▸ DASHBOARD", value="tab-dashboard",
                                style=TAB_STYLE, selected_style=TAB_SELECTED),
                    ],
                    style={"background": "transparent", "border": "none", "marginBottom": "32px"},
                    colors={"border": C["border"], "primary": C["accent"], "background": "transparent"},
                ),
                html.Div(id="tab-content"),
            ]
        )
    ]
)


# ─── Tab router ───────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab-forecast":
        return forecast_layout()
    return dashboard_layout()


# ══════════════════════════════════════════════════════════════════════════════
#  FORECASTING TAB
# ══════════════════════════════════════════════════════════════════════════════
def forecast_layout():
    return html.Div([
        # Hero
        html.Div(style={"marginBottom": "40px"}, children=[
            html.H1("Stock Price Forecasting",
                    style={"fontSize": "44px", "fontWeight": "800", "margin": "0 0 8px",
                           "color": C["accent"], "letterSpacing": "0.5px"}),
            html.P("LSTM neural network predictions · MLflow-tracked best models",
                   style={"color": C["muted"], "fontSize": "18px", "margin": "0", "fontWeight": "500"}),
        ]),

        # Controls
        html.Div(style=CARD, children=[
            section_title("Control Panel"),
            html.Div(style={"display": "grid", "gridTemplateColumns": "2fr 1fr 1fr 1fr", "gap": "18px", "alignItems": "end"},
                     children=[
                         html.Div([
                             html.Label("Select Stock", style=LABEL),
                             dcc.Dropdown(
                                 id="fc-stock",
                                 options=[{"label": f"{s['ticker']}  —  {s['name']}", "value": s['ticker']} for s in STOCKS],
                                 value=STOCKS[0]["ticker"],
                                 clearable=False,
                                 className="stock-dropdown",
                                 style=DD_STYLE,
                             ),
                         ]),
                         html.Div([
                             html.Label("Forecast Days (1–40)", style=LABEL),
                             dcc.Input(id="fc-days", type="number", value=7, min=1, max=40,
                                       style=INPUT_STYLE),
                         ]),
                         html.Div([
                             html.Label("\u00a0", style=LABEL),
                             html.Button("◈ Visualize", id="fc-vis-btn", n_clicks=0, style=BTN_PRIMARY),
                         ]),
                         html.Div([
                             html.Label("\u00a0", style=LABEL),
                             html.Button("⬡ Predict Ahead", id="fc-pred-btn", n_clicks=0, style=BTN_SUCCESS),
                         ]),
                     ]),

            html.Div(style={"marginTop": "35px", "padding": "0 10px"}, children=[
                html.Label("Quick-select forecast horizon", style=LABEL),
                dcc.Slider(
                    id="fc-slider",
                    min=1, max=40, step=1, value=7,
                    marks={
                        i: {"label": str(i), "style": {"color": C["muted"], "fontSize": "16px", "fontWeight": "700"}} 
                        for i in [1, 10, 20, 30, 40]
                    },
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ]),
        ]),

        # Error / info banner
        html.Div(id="fc-banner", style={"display": "none"}),

        # Chart
        html.Div(style=CARD, children=[
            section_title("Prediction Chart"),
            dcc.Graph(id="fc-chart", style={"minHeight": "550px"},
                     config={"displayModeBar": False}),
        ]),

        # Stats row
        html.Div(id="fc-stats", style={"display": "none"},
                 children=[
                     html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)", "gap": "20px"},
                              children=[
                                  stat_card("fc-stat-ticker",  "Stock"),
                                  stat_card("fc-stat-last",    "Last Close",    color=C["accent2"]),
                                  stat_card("fc-stat-days",    "Forecast Days", color=C["gold"]),
                                  stat_card("fc-stat-pred",    "Predicted End", color=C["purple"]),
                              ])
                 ]),
    ])


def stat_card(elem_id, label, color=None):
    return html.Div(style={
        **CARD,
        "textAlign": "center",
        "marginBottom": "0",
        "borderTop": f"4px solid {color or C['accent']}",
    }, children=[
        html.Div(label, style={**LABEL, "marginBottom": "16px", "fontSize": "13px"}),
        html.Div("—", id=elem_id, style={
            "fontSize": "48px", "fontWeight": "800",
            "color": color or C["accent"],
            "fontFamily": "'Inter', sans-serif",
            "lineHeight": "1.2",
        }),
    ])


@app.callback(
    Output("fc-days", "value"),
    Input("fc-slider", "value"),
    prevent_initial_call=True,
)
def sync_slider_to_input(val):
    return val


@app.callback(
    [Output("fc-chart", "figure"),
     Output("fc-banner", "children"),
     Output("fc-banner", "style"),
     Output("fc-stats", "style"),
     Output("fc-stat-ticker", "children"),
     Output("fc-stat-last", "children"),
     Output("fc-stat-days", "children"),
     Output("fc-stat-pred", "children")],
    [Input("fc-vis-btn", "n_clicks"),
     Input("fc-pred-btn", "n_clicks")],
    [State("fc-stock", "value"),
     State("fc-days", "value")],
    prevent_initial_call=True,
)
def update_forecast(vis_clicks, pred_clicks, ticker, n_days):
    empty_stats = ("—", "—", "—", "—")
    hidden = {"display": "none"}
    banner_err = lambda msg: (
        {}, msg, {
            "background": "rgba(211,47,47,0.12)", "border": f"2px solid {C['red']}",
            "borderRadius": "8px", "padding": "16px 22px", "marginBottom": "22px",
            "color": C["red"], "fontSize": "16px", "display": "block", "fontWeight": "600",
        },
        hidden, *empty_stats,
    )

    if not ticker:
        return banner_err("⚠ Please select a stock.")

    stock = next((s for s in STOCKS if s["ticker"] == ticker), None)
    if not stock:
        return banner_err(f"⚠ Unknown ticker: {ticker}")

    ctx = callback_context
    if not ctx.triggered:
        return go.Figure(), "", hidden, hidden, *empty_stats

    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    n_days = int(n_days or 7)

    try:
        model = load_model_for_stock(ticker, stock["name"])
        data  = load_data_for_stock(ticker)

        if btn == "fc-vis-btn":
            actuals, preds, dates = get_validation_actuals_and_preds(model, data)
            fig = styled_fig()
            fig.add_trace(go.Scatter(x=dates, y=actuals, name="Actual",
                                     line=dict(color=C["accent2"], width=2.5)))
            fig.add_trace(go.Scatter(x=dates, y=preds, name="Model Prediction",
                                     line=dict(color=C["accent"], width=2.5, dash="dot")))
            fig.update_layout(title=dict(text=f"{stock['name']}  ·  Validation", font=dict(size=22, color=C["text"])))
            info_style = {
                "background": "rgba(0,102,204,0.12)", "border": f"2px solid {C['accent']}",
                "borderRadius": "8px", "padding": "16px 22px", "marginBottom": "22px",
                "color": C["accent"], "fontSize": "16px", "display": "block", "fontWeight": "600",
            }
            return (fig, f"✓ Loaded {len(actuals)} validation points for {ticker}",
                    info_style, hidden, *empty_stats)

        # Predict ahead
        if n_days > 40:
            return banner_err("⚠ Max 40 days. Please reduce the horizon.")
        if n_days < 1:
            return banner_err("⚠ Minimum 1 day.")

        preds_fut, fut_dates, warn = predict_n_days_ahead(model, data, n_days=n_days)
        if warn:
            return banner_err(warn)

        actuals, val_preds, val_dates = get_validation_actuals_and_preds(model, data)
        val_dates_dt = pd.to_datetime(val_dates, errors="coerce")
        fut_dates_dt = pd.to_datetime(fut_dates, errors="coerce")
        if pd.isna(val_dates_dt).any() or pd.isna(fut_dates_dt).any():
            return banner_err("Error: Could not parse forecast dates.")

        last_date = val_dates_dt[-1]
        fig = styled_fig()
        fig.add_trace(go.Scatter(
            x=val_dates_dt, y=actuals, name="Actual Price",
            line=dict(color=C["accent2"], width=2),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.04)",
        ))
        fig.add_trace(go.Scatter(
            x=val_dates_dt, y=val_preds, name="Model Fit",
            line=dict(color=C["accent"], width=1.5, dash="dot"),
        ))
        bridge_x = [last_date, fut_dates_dt[0]]
        bridge_y = [actuals[-1], preds_fut[0]]
        fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, showlegend=False,
                                  line=dict(color=C["gold"], width=2, dash="dot")))
        fig.add_trace(go.Scatter(
            x=fut_dates_dt, y=preds_fut,
            name=f"Forecast ({n_days}d)",
            mode="lines+markers",
            line=dict(color=C["gold"], width=2.5),
            marker=dict(size=6, color=C["gold"], line=dict(color=C["bg"], width=1.5)),
        ))
        
        marker_x = pd.Timestamp(last_date).to_pydatetime()
        fig.add_shape(
            type="line",
            x0=marker_x,
            x1=marker_x,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=C["gold"], width=1, dash="dash"),
        )
        fig.add_annotation(
            x=marker_x,
            y=1,
            xref="x",
            yref="paper",
            text="▶ forecast",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color=C["gold"], size=11),
        )
        fig.update_layout(title=dict(text=f"{stock['name']}  ·  {n_days}-Day Forecast",
                                     font=dict(size=22, color=C["text"])))

        last_close = actuals[-1]
        end_pred   = preds_fut[-1]
        chg = (end_pred - last_close) / last_close * 100
        chg_str = f"${end_pred:.2f}  ({chg:+.1f}%)"

        warn_style = {
            "background": "rgba(255,152,0,0.12)", "border": f"2px solid {C['gold']}",
            "borderRadius": "8px", "padding": "16px 22px", "marginBottom": "22px",
            "color": C["gold"], "fontSize": "16px", "display": "block", "fontWeight": "600",
        }
        banner_msg = ""
        banner_sty = hidden
        if n_days >= 30:
            banner_msg = "⚠ Predictions beyond 30 days carry higher uncertainty."
            banner_sty = warn_style

        return (fig, banner_msg, banner_sty,
                {"display": "block"},
                ticker,
                f"${last_close:.2f}",
                str(n_days),
                chg_str)

    except Exception as e:
        traceback.print_exc()
        return banner_err(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD TAB
# ══════════════════════════════════════════════════════════════════════════════
def dashboard_layout():
    return html.Div([
        html.Div(style={"marginBottom": "40px"}, children=[
            html.H1("Stock Analysis Dashboard",
                    style={"fontSize": "44px", "fontWeight": "800", "margin": "0 0 8px",
                           "color": C["accent"], "letterSpacing": "0.5px"}),
            html.P("Technical analysis · Historical data · Risk metrics",
                   style={"color": C["muted"], "fontSize": "18px", "margin": "0", "fontWeight": "500"}),
        ]),

        # Controls
        html.Div(style=CARD, children=[
            section_title("Parameters"),
            html.Div(style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "22px"},
                     children=[
                         html.Div([
                             html.Label("Stock", style=LABEL),
                             dcc.Dropdown(
                                 id="db-stock",
                                 options=[{"label": f"{s['ticker']}  —  {s['name']}", "value": s['ticker']} for s in STOCKS],
                                 value=STOCKS[0]["ticker"],
                                 clearable=False, className="stock-dropdown", style=DD_STYLE,
                             ),
                         ]),
                         html.Div([
                             html.Label("Time Period", style=LABEL),
                             dcc.Dropdown(
                                 id="db-period",
                                 options=[
                                     {"label": "1 Month",  "value": "1mo"},
                                     {"label": "3 Months", "value": "3mo"},
                                     {"label": "6 Months", "value": "6mo"},
                                     {"label": "1 Year",   "value": "1y"},
                                     {"label": "2 Years",  "value": "2y"},
                                     {"label": "5 Years",  "value": "5y"},
                                     {"label": "20 Years", "value": "20y"},
                                 ],
                                 value="1y",
                                 clearable=False, style=DD_STYLE,
                             ),
                         ]),
                     ]),
        ]),

        # KPI cards row (dynamic)
        html.Div(id="db-kpis", style={"marginBottom": "25px"}),

        # Charts grid
        html.Div(style=CARD, children=[
            section_title("Candlestick · Price Action"),
            dcc.Graph(id="db-candle", config={"displayModeBar": False}, style={"height": "450px"}),
        ]),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "22px"}, children=[
            html.Div(style=CARD, children=[
                section_title("Price + Moving Averages + Bollinger Bands"),
                dcc.Graph(id="db-bollinger", config={"displayModeBar": False}, style={"height": "400px"}),
            ]),
            html.Div(style=CARD, children=[
                section_title("Volume Analysis"),
                dcc.Graph(id="db-volume", config={"displayModeBar": False}, style={"height": "400px"}),
            ]),
        ]),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "22px"}, children=[
            html.Div(style=CARD, children=[
                section_title("Daily Returns Distribution"),
                dcc.Graph(id="db-returns-dist", config={"displayModeBar": False}, style={"height": "320px"}),
            ]),
            html.Div(style=CARD, children=[
                section_title("Annualized Rolling Volatility"),
                dcc.Graph(id="db-volatility", config={"displayModeBar": False}, style={"height": "320px"}),
            ]),
        ]),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
            html.Div(style=CARD, children=[
                section_title("Cumulative Return"),
                dcc.Graph(id="db-cumret", config={"displayModeBar": False}, style={"height": "320px"}),
            ]),
            html.Div(style=CARD, children=[
                section_title("Drawdown from Peak"),
                dcc.Graph(id="db-drawdown", config={"displayModeBar": False}, style={"height": "320px"}),
            ]),
        ]),
    ])


@app.callback(
    [Output("db-kpis",         "children"),
     Output("db-candle",       "figure"),
     Output("db-bollinger",    "figure"),
     Output("db-volume",       "figure"),
     Output("db-returns-dist","figure"),
     Output("db-volatility",  "figure"),
     Output("db-cumret",       "figure"),
     Output("db-drawdown",     "figure")],
    [Input("db-stock",  "value"),
     Input("db-period", "value")],
)
def update_dashboard(ticker, period):
    empty = go.Figure()
    empty.update_layout(**PLOT_LAYOUT)

    df = get_stock_data(ticker, period)
    if df is None or df.empty:
        return html.Div("No data"), *([empty] * 7)

    stock = next((s for s in STOCKS if s["ticker"] == ticker), {"name": ticker})
    name  = stock["name"]

    # ── KPI cards ─────────────────────────────────────────────────────────────
    last  = df['Close'].iloc[-1]
    prev  = df['Close'].iloc[-2]
    chg   = (last - prev) / prev * 100
    high  = df['High'].max()
    low   = df['Low'].min()
    vol_m = df['Volume'].mean() / 1e6
    ret   = df['CumReturn'].iloc[-1] * 100
    vol_ann = df['Volatility'].iloc[-1] * 100
    max_dd = df['Drawdown'].min() * 100

    def kpi(label, value, sub="", color=C["accent"]):
        return html.Div(style={
            **CARD,
            "marginBottom": "0",
            "borderTop": f"4px solid {color}",
            "textAlign": "center",
        }, children=[
            html.Div(label, style={**LABEL, "marginBottom": "12px", "fontSize": "13px"}),
            html.Div(value, style={"fontSize": "44px", "fontWeight": "800", "color": color,
                                    "fontFamily": "'Inter', sans-serif", "lineHeight": "1.2"}),
            html.Div(sub, style={"fontSize": "14px", "color": C["muted"], "marginTop": "8px", "fontWeight": "600"}),
        ])

    kpis = html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(7,1fr)", "gap": "15px"}, children=[
        kpi("Last Close",  f"${last:.2f}",
            f"{chg:+.2f}%", color=C["accent2"] if chg >= 0 else C["red"]),
        kpi("Period High", f"${high:.2f}",  color=C["accent2"]),
        kpi("Period Low",  f"${low:.2f}",   color=C["red"]),
        kpi("Avg Volume",  f"{vol_m:.1f}M", color=C["muted"]),
        kpi("Cum Return",  f"{ret:+.1f}%",  color=C["accent2"] if ret >= 0 else C["red"]),
        kpi("Volatility",  f"{vol_ann:.1f}%", "annualised", color=C["gold"]),
        kpi("Max Drawdown",f"{max_dd:.1f}%", "from peak",  color=C["red"]),
    ])

    # ── Candlestick ────────────────────────────────────────────────────────────
    fig_candle = styled_fig()
    fig_candle.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color=C["accent2"], decreasing_line_color=C["red"],
        increasing_fillcolor=C["accent2"], decreasing_fillcolor=C["red"],
        name=ticker,
    ))
    fig_candle.update_layout(xaxis_rangeslider_visible=False,
                              title=dict(text=f"{name}  ·  Candlestick", font=dict(size=22, color=C["text"])))

    # ── Bollinger + SMAs ────────────────────────────────────────────────────────
    fig_bb = styled_fig()
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Upper'],
                                 line=dict(color="rgba(0,102,204,0.25)", width=1),
                                 name="Upper Band", showlegend=True))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Lower'],
                                 line=dict(color="rgba(0,102,204,0.25)", width=1),
                                 fill="tonexty", fillcolor="rgba(0,102,204,0.07)",
                                 name="Lower Band"))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                                 line=dict(color=C["accent2"], width=2.5), name="Close"))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'],
                                 line=dict(color=C["gold"], width=2, dash="dash"), name="SMA 20"))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'],
                                 line=dict(color=C["purple"], width=2, dash="dot"), name="SMA 50"))
    fig_bb.update_layout(title=dict(text="Bollinger Bands", font=dict(size=22, color=C["text"])))

    # ── Volume ────────────────────────────────────────────────────────────────
    colors_vol = [C["accent2"] if c >= o else C["red"]
                  for c, o in zip(df['Close'], df['Open'])]
    fig_vol = styled_fig()
    fig_vol.add_trace(go.Bar(x=df['Date'], y=df['Volume'] / 1e6,
                              marker_color=colors_vol, name="Volume (M)",
                              marker_line_width=0))
    fig_vol.update_layout(title=dict(text="Volume (millions)", font=dict(size=22, color=C["text"])),
                          bargap=0.1)

    # ── Returns distribution ──────────────────────────────────────────────────
    rets_pct = df['Returns'].dropna() * 100
    fig_dist = styled_fig()
    fig_dist.add_trace(go.Histogram(
        x=rets_pct, nbinsx=50,
        marker_color=C["accent"],
        marker_line_color=C["bg"],
        marker_line_width=0.5,
        opacity=0.85,
        name="Returns (%)",
    ))
    # Add normal curve overlay
    mu, sigma = rets_pct.mean(), rets_pct.std()
    x_range = np.linspace(rets_pct.min(), rets_pct.max(), 200)
    normal_y = (len(rets_pct) * (rets_pct.max() - rets_pct.min()) / 50) * \
               (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    fig_dist.add_trace(go.Scatter(x=x_range, y=normal_y,
                                   line=dict(color=C["gold"], width=2.5), name="Normal fit"))
    fig_dist.update_layout(
        title=dict(text="Daily Returns Distribution (%)", font=dict(size=22, color=C["text"])),
        xaxis_title="Return (%)",
        bargap=0.05,
    )

    # ── Volatility ────────────────────────────────────────────────────────────
    fig_vola = styled_fig()
    fig_vola.add_trace(go.Scatter(
        x=df['Date'], y=df['Volatility'] * 100,
        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
        line=dict(color=C["gold"], width=2.5), name="Volatility (%)",
    ))
    fig_vola.update_layout(title=dict(text="Annualised Rolling Volatility (20d)", font=dict(size=22, color=C["text"])))

    # ── Cumulative return ─────────────────────────────────────────────────────
    fig_cum = styled_fig()
    cum_pct = df['CumReturn'] * 100
    colors_cum = [C["accent2"] if v >= 0 else C["red"] for v in cum_pct]
    fig_cum.add_trace(go.Scatter(
        x=df['Date'], y=cum_pct,
        fill="tozeroy",
        fillcolor="rgba(0,200,83,0.06)",
        line=dict(color=C["accent2"], width=2.5), name="Cum. Return (%)",
    ))
    fig_cum.add_hline(y=0, line=dict(color=C["muted"], width=1, dash="dot"))
    fig_cum.update_layout(title=dict(text="Cumulative Return (%)", font=dict(size=22, color=C["text"])))

    # ── Drawdown ──────────────────────────────────────────────────────────────
    fig_dd = styled_fig()
    fig_dd.add_trace(go.Scatter(
        x=df['Date'], y=df['Drawdown'] * 100,
        fill="tozeroy", fillcolor="rgba(211,47,47,0.08)",
        line=dict(color=C["red"], width=2.5), name="Drawdown (%)",
    ))
    fig_dd.update_layout(title=dict(text="Drawdown from Peak (%)", font=dict(size=22, color=C["text"])))

    return kpis, fig_candle, fig_bb, fig_vol, fig_dist, fig_vola, fig_cum, fig_dd


if __name__ == "__main__":
    app.run(debug=True, port=8050)