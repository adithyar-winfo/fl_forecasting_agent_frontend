import streamlit as st
import requests
import uuid
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
from io import StringIO
from pathlib import Path

# -------------------------------
# Config
# -------------------------------
REMOTE_CHAT_URL = "https://fl-forecasting-agent-backend.onrender.com/agent/chat"
REMOTE_REFRESH_AGENT_URL = "https://fl-forecasting-agent-backend.onrender.com/agent/refresh_agent"
LOCAL_CHAT_URL = "http://127.0.0.1:8010/agent/chat"
LOCAL_REFRESH_AGENT_URL = "http://127.0.0.1:8010/agent/refresh_agent"


def _resolve_default_api_urls() -> tuple[str, str]:
    env_chat = os.getenv("CHAT_URL")
    env_refresh = os.getenv("REFRESH_AGENT_URL")
    if env_chat or env_refresh:
        return (
            env_chat or REMOTE_CHAT_URL,
            env_refresh or REMOTE_REFRESH_AGENT_URL,
        )

    # If local chat backend is running, prefer it by default.
    try:
        health = requests.get("http://127.0.0.1:8010/health", timeout=1.2)
        if health.ok:
            return LOCAL_CHAT_URL, LOCAL_REFRESH_AGENT_URL
    except requests.RequestException:
        pass

    return REMOTE_CHAT_URL, REMOTE_REFRESH_AGENT_URL


CHAT_URL, REFRESH_AGENT_URL = _resolve_default_api_urls()
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "90"))

APP_DIR = Path(__file__).resolve().parent
TAB_ICON_PATH = APP_DIR / "circlelogo.png"
SIDEBAR_LOGO_PATH = APP_DIR / "winfo-new-logo.png"

st.set_page_config(
    page_title="AI Agent",
    page_icon=str(TAB_ICON_PATH) if TAB_ICON_PATH.exists() else None,
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .block-container {
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] .stImage {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": None, "-": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _clean_x_labels(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\*\*", "", regex=True)
        .str.replace(r"`", "", regex=True)
        .str.strip()
        .str.strip("'\"")
        .str.strip()
    )


def _parse_dates_with_known_formats(series: pd.Series) -> pd.Series | None:
    # Use deterministic formats to avoid pandas fallback warnings.
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y"]
    best = None
    best_ratio = 0.0

    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        ratio = parsed.notna().mean()
        if ratio > best_ratio:
            best_ratio = ratio
            best = parsed

    if best is not None and best_ratio >= 0.6:
        return best

    return None


def _find_date_axis_column(df: pd.DataFrame) -> str | None:
    best_col = None
    best_score = 0.0

    for c in df.columns:
        cleaned = _clean_x_labels(df[c])
        parsed = _parse_dates_with_known_formats(cleaned)
        if parsed is None:
            continue

        ratio = parsed.notna().mean()
        if ratio < 0.6:
            continue

        name = str(c).lower()
        keyword_bonus = 0.1 if any(k in name for k in ["date", "arrival", "eta"]) else 0.0
        score = ratio + keyword_bonus

        if score > best_score:
            best_score = score
            best_col = c

    return best_col


def _format_backend_error(response: requests.Response) -> tuple[str, str | None]:
    status = response.status_code
    body = response.text or ""
    content_type = (response.headers.get("content-type") or "").lower()

    # Cloudflare-origin outage handling.
    if status == 520 and ("cloudflare" in body.lower() or "error code 520" in body.lower()):
        ray_id_match = re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", body, re.IGNORECASE)
        ray_id = ray_id_match.group(1).strip() if ray_id_match else None
        return (
            "Backend is temporarily unavailable (Cloudflare 520). Please retry in a few minutes.",
            ray_id,
        )

    if "application/json" in content_type:
        try:
            err_json = response.json()
            detail = err_json.get("detail") or err_json.get("message") or str(err_json)
            return f"Request failed ({status}): {detail}", None
        except Exception:
            pass

    # Prevent dumping huge HTML pages into chat UI.
    if "<html" in body.lower():
        return f"Request failed ({status}): Received an HTML error page from backend.", None

    compact = re.sub(r"\s+", " ", body).strip()
    if len(compact) > 300:
        compact = f"{compact[:300]}..."
    return f"Request failed ({status}): {compact}", None


def _extract_markdown_tables(text: str) -> list[pd.DataFrame]:
    if not text:
        return []

    lines = text.splitlines()
    tables: list[pd.DataFrame] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current.append(stripped)
        else:
            if current:
                tables.append(_parse_markdown_table_block(current))
                current = []

    if current:
        tables.append(_parse_markdown_table_block(current))

    return [df for df in tables if df is not None and not df.empty]


def _parse_markdown_table_block(rows: list[str]) -> pd.DataFrame | None:
    if len(rows) < 2:
        return None

    # Drop markdown separator row like |---|---|
    normalized_rows = [r.strip().strip("|") for r in rows]
    if len(normalized_rows) >= 2 and re.fullmatch(r"\s*:?-+:?\s*(\|\s*:?-+:?\s*)*", normalized_rows[1]):
        normalized_rows.pop(1)

    if len(normalized_rows) < 2:
        return None

    csv_like = "\n".join(normalized_rows)
    try:
        df = pd.read_csv(StringIO(csv_like), sep="|", engine="python")
    except Exception:
        return None

    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def _build_line_chart(df: pd.DataFrame) -> tuple[go.Figure | None, pd.DataFrame | None, str | None]:
    if df is None or df.empty:
        return None, None, None

    working = df.copy()
    columns = list(working.columns)
    lower = {c: c.lower() for c in columns}

    # Forecast graphs must use date on the x-axis.
    x_col = _find_date_axis_column(working)
    if x_col is None:
        return None, None, None

    # Remove aggregate rows (e.g., Total / Grand Total / Overall / Sum)
    # across all text columns, not only the chosen x-axis.
    object_cols = [c for c in columns if working[c].dtype == object]
    if object_cols:
        aggregate_mask = pd.Series(False, index=working.index)
        for c in object_cols:
            cleaned_col = _clean_x_labels(working[c]).str.lower()
            aggregate_mask = aggregate_mask | cleaned_col.str.contains(
                r"\b(?:total|grand\s+total|overall|sum)\b",
                regex=True,
                na=False,
            )
        working = working.loc[~aggregate_mask].copy()
    else:
        x_series = _clean_x_labels(working[x_col]).str.lower()
        aggregate_mask = x_series.str.contains(
            r"\b(?:total|grand\s+total|overall|sum)\b",
            regex=True,
            na=False,
        )
        working = working.loc[~aggregate_mask].copy()

    if working.empty:
        return None, None, None

    y_candidates = []
    for c in columns:
        if c == x_col:
            continue
        numeric = _to_numeric(working[c])
        if numeric.notna().sum() >= 2:
            working[c] = numeric
            y_candidates.append(c)

    preferred_y = [
        c for c in y_candidates
        if any(k in lower[c] for k in ["predicted", "forecast", "demand"])
    ]
    y_cols = preferred_y if preferred_y else y_candidates

    if not y_cols:
        return None, None, None

    # Date axis is mandatory.
    working[x_col] = _clean_x_labels(working[x_col])
    x_dt = _parse_dates_with_known_formats(working[x_col])
    if x_dt is None or x_dt.notna().sum() < 2:
        return None, None, None

    working[x_col] = x_dt
    working = working.dropna(subset=[x_col]).sort_values(x_col)

    # Prefer explicit Actual vs Forecast rendering when table contains
    # ordered/actual-like and predicted/forecast-like numeric columns.
    actual_like = [c for c in y_candidates if any(k in lower[c] for k in ["actual", "ordered"])]
    forecast_like = [c for c in y_candidates if any(k in lower[c] for k in ["predicted", "forecast", "expected"])]
    if actual_like and forecast_like:
        actual_col = actual_like[0]
        forecast_col = forecast_like[0]

        # Avoid selecting the same column for both roles.
        if actual_col != forecast_col:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=working[x_col],
                    y=working[actual_col],
                    mode="lines+markers",
                    name=f"Actual ({actual_col})",
                    line={"dash": "solid", "color": "#111111"},
                    marker={"color": "#111111"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=working[x_col],
                    y=working[forecast_col],
                    mode="lines+markers",
                    name=f"Forecast ({forecast_col})",
                    line={"dash": "dot", "color": "#b22222"},
                    marker={"color": "#b22222"},
                )
            )
            fig.update_layout(
                height=520,
                title="Actual vs Forecast Trend",
                xaxis_title="Date",
                yaxis_title=f"{actual_col} / {forecast_col}",
                legend={"orientation": "h", "y": -0.2},
            )
            return fig, working, "Actual vs Forecast Trend"

    # If there is one forecast column and a unit-like column, pivot into multiline chart.
    if len(y_cols) == 1:
        unit_col = None
        for c in columns:
            if c == x_col or c == y_cols[0]:
                continue
            if any(k in lower[c] for k in ["sku", "unit", "item", "product", "category"]):
                unit_col = c
                break

        if unit_col is not None:
            pivot_df = (
                working[[x_col, unit_col, y_cols[0]]]
                .dropna(subset=[x_col, unit_col, y_cols[0]])
                .pivot_table(index=x_col, columns=unit_col, values=y_cols[0], aggfunc="sum")
                .sort_index()
            )

            if not pivot_df.empty and len(pivot_df.columns) >= 2:
                fig = go.Figure()
                for series_name in pivot_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=pivot_df.index,
                            y=pivot_df[series_name],
                            mode="lines+markers",
                            name=str(series_name),
                        )
                    )

                fig.update_layout(
                    height=520,
                    title="Forecast Trend",
                    xaxis_title=x_col,
                    yaxis_title=y_cols[0],
                    legend={"orientation": "h", "y": -0.2},
                )
                return fig, working, "Forecast Trend"

    fig = go.Figure()
    for c in y_cols:
        fig.add_trace(
            go.Scatter(
                x=working[x_col],
                y=working[c],
                mode="lines+markers",
                name=c,
            )
        )

    y_axis_title = y_cols[0] if len(y_cols) == 1 else " / ".join(y_cols)

    fig.update_layout(
        height=520,
        title="Forecast Trend",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        legend={"orientation": "h", "y": -0.2},
    )

    return fig, working, "Forecast Trend"


def _build_actual_forecast_date_chart(df: pd.DataFrame) -> tuple[go.Figure | None, pd.DataFrame | None, str | None]:
    """Strict forecast chart builder: requires date x-axis and explicit actual/forecast columns."""
    if df is None or df.empty:
        return None, None, None

    working = df.copy()
    columns = list(working.columns)
    lower = {c: c.lower() for c in columns}

    x_col = _find_date_axis_column(working)
    if x_col is None:
        return None, None, None

    object_cols = [c for c in columns if working[c].dtype == object]
    if object_cols:
        aggregate_mask = pd.Series(False, index=working.index)
        for c in object_cols:
            cleaned_col = _clean_x_labels(working[c]).str.lower()
            aggregate_mask = aggregate_mask | cleaned_col.str.contains(
                r"\b(?:total|grand\s+total|overall|sum|subtotal)\b",
                regex=True,
                na=False,
            )
        working = working.loc[~aggregate_mask].copy()
    if working.empty:
        return None, None, None

    y_candidates = []
    for c in columns:
        if c == x_col:
            continue
        numeric = _to_numeric(working[c])
        if numeric.notna().sum() >= 2:
            working[c] = numeric
            y_candidates.append(c)

    actual_like = [c for c in y_candidates if any(k in lower[c] for k in ["actual", "ordered", "baseline"])]
    forecast_like = [c for c in y_candidates if any(k in lower[c] for k in ["predicted", "forecast", "expected"])]
    if not actual_like or not forecast_like:
        return None, None, None

    actual_col = actual_like[0]
    forecast_col = next((c for c in forecast_like if c != actual_col), None)
    if forecast_col is None:
        return None, None, None

    working[x_col] = _clean_x_labels(working[x_col])
    x_dt = _parse_dates_with_known_formats(working[x_col])
    if x_dt is None or x_dt.notna().sum() < 2:
        return None, None, None

    working[x_col] = x_dt
    working = working.dropna(subset=[x_col]).sort_values(x_col)
    if working.empty:
        return None, None, None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=working[x_col],
            y=working[actual_col],
            mode="lines+markers",
            name=f"Actual ({actual_col})",
            line={"dash": "solid", "color": "#111111"},
            marker={"color": "#111111"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=working[x_col],
            y=working[forecast_col],
            mode="lines+markers",
            name=f"Forecast ({forecast_col})",
            line={"dash": "dot", "color": "#b22222"},
            marker={"color": "#b22222"},
        )
    )
    fig.update_layout(
        height=520,
        title="Actual vs Forecast Trend",
        xaxis_title="Date",
        yaxis_title=f"{actual_col} / {forecast_col}",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig, working, "Actual vs Forecast Trend"


def _build_labor_table_chart(df: pd.DataFrame) -> tuple[go.Figure | None, pd.DataFrame | None, str | None]:
    """Fallback chart for labor summary tables that are category-based (no date axis)."""
    if df is None or df.empty:
        return None, None, None

    working = df.copy()
    columns = list(working.columns)
    lower = {c: c.lower() for c in columns}

    category_col = next(
        (c for c in columns if any(k in lower[c] for k in ["process", "shift", "operation", "category"])),
        None,
    )
    if category_col is None:
        return None, None, None

    # Remove subtotal/total rows for categorical labor bars.
    cat = _clean_x_labels(working[category_col]).str.lower()
    total_mask = cat.str.contains(r"\b(?:total|subtotal|grand\s+total|overall|sum)\b", regex=True, na=False)
    working = working.loc[~total_mask].copy()
    if working.empty:
        return None, None, None

    preferred = ["total labor", "labor hours", "required daily staffing", "required headcount", "avg daily labor"]
    value_col = None
    for key in preferred:
        value_col = next((c for c in columns if key in lower[c]), None)
        if value_col:
            break
    if value_col is None:
        numeric_cols = [c for c in columns if c != category_col and _to_numeric(working[c]).notna().sum() >= 2]
        value_col = numeric_cols[0] if numeric_cols else None
    if value_col is None:
        return None, None, None

    working[value_col] = _to_numeric(working[value_col])
    working[category_col] = _clean_x_labels(working[category_col])
    working = working.dropna(subset=[category_col, value_col])
    if working.empty:
        return None, None, None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=working[category_col],
            y=working[value_col],
            marker={"color": "#1f77b4"},
            name=value_col,
        )
    )
    fig.update_layout(
        height=500,
        title="Labor Forecast by Category",
        xaxis_title=category_col,
        yaxis_title=value_col,
    )
    return fig, working, "Labor Forecast by Category"


def _build_labor_daily_process_chart(df: pd.DataFrame) -> tuple[go.Figure | None, pd.DataFrame | None, str | None]:
    """Labor chart with date on X, numeric values on primary Y, and process labels on secondary Y."""
    if df is None or df.empty:
        return None, None, None

    working = df.copy()
    columns = list(working.columns)
    lower = {c: c.lower() for c in columns}

    x_col = _find_date_axis_column(working)
    process_col = next((c for c in columns if "process" in lower[c]), None)
    if x_col is None or process_col is None:
        return None, None, None

    preferred_values = [
        "daily labor hours",
        "labor hours",
        "required daily staffing",
        "required headcount",
        "avg daily labor",
        "driver units",
        "workload",
    ]
    value_col = None
    for key in preferred_values:
        value_col = next((c for c in columns if key in lower[c]), None)
        if value_col is not None:
            break

    if value_col is None:
        numeric_cols = [c for c in columns if c not in (x_col, process_col) and _to_numeric(working[c]).notna().sum() >= 2]
        value_col = numeric_cols[0] if numeric_cols else None
    if value_col is None:
        return None, None, None

    working[x_col] = _clean_x_labels(working[x_col])
    parsed_dates = _parse_dates_with_known_formats(working[x_col])
    if parsed_dates is None or parsed_dates.notna().sum() < 2:
        return None, None, None

    working[x_col] = parsed_dates
    working[process_col] = _clean_x_labels(working[process_col])
    working[value_col] = _to_numeric(working[value_col])
    working = working.dropna(subset=[x_col, process_col, value_col])

    process_mask = ~working[process_col].str.lower().str.contains(
        r"\b(?:total|subtotal|grand\s+total|overall|sum)\b",
        regex=True,
        na=False,
    )
    working = working.loc[process_mask].copy()
    if working.empty:
        return None, None, None

    grouped = (
        working[[x_col, process_col, value_col]]
        .groupby([x_col, process_col], as_index=False)[value_col]
        .sum()
        .sort_values([x_col, process_col])
    )
    if grouped.empty:
        return None, None, None

    processes = grouped[process_col].dropna().astype(str).unique().tolist()
    if not processes:
        return None, None, None

    fig = go.Figure()
    for idx, proc in enumerate(processes):
        proc_df = grouped[grouped[process_col] == proc]
        if proc_df.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=proc_df[x_col],
                y=proc_df[value_col],
                mode="lines+markers",
                name=proc,
                yaxis="y",
            )
        )

        # Invisible traces anchor process ticks on secondary axis without cluttering the chart.
        fig.add_trace(
            go.Scatter(
                x=proc_df[x_col],
                y=[idx] * len(proc_df),
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
                yaxis="y2",
            )
        )

    fig.update_layout(
        height=520,
        title="Labor Trend by Process (Daily)",
        xaxis_title="Date",
        yaxis={"title": value_col},
        yaxis2={
            "title": "Operational Process",
            "overlaying": "y",
            "side": "right",
            "tickmode": "array",
            "tickvals": list(range(len(processes))),
            "ticktext": processes,
            "showgrid": False,
            "zeroline": False,
        },
        legend={"orientation": "h", "y": -0.2},
    )

    return fig, grouped, "Labor Trend by Process (Daily)"


def _is_labor_context(prompt: str, data_type: str | None, reply: str) -> bool:
    text = f"{prompt} {data_type or ''} {reply}".lower()
    return "labor" in text or "labour" in text or "headcount" in text or "staffing" in text


def _is_forecast_intent(prompt: str, data_type: str | None, reply: str) -> bool:
    text = f"{prompt} {data_type or ''} {reply}".lower()
    return "forecast" in text or "predicted" in text


def _normalize_series_data(series_data: pd.DataFrame, x_field: str) -> pd.DataFrame:
    if series_data.empty or x_field not in series_data.columns or "value" not in series_data.columns:
        return pd.DataFrame(columns=[x_field, "value"])

    working = series_data.copy()
    working["value"] = _to_numeric(working["value"])

    if x_field == "date":
        working["date"] = _clean_x_labels(working["date"])
        parsed = _parse_dates_with_known_formats(working["date"])
        if parsed is None:
            return pd.DataFrame(columns=["date", "value"])
        working["date"] = parsed
        working = working.dropna(subset=["date", "value"]).sort_values("date")
        return working[["date", "value"]]

    if x_field == "category":
        working["category"] = _clean_x_labels(working["category"]) if "category" in working.columns else ""
        working = working.dropna(subset=["value"])
        working = working[working["category"].astype(str).str.strip() != ""]
        return working[["category", "value"]]

    return pd.DataFrame(columns=[x_field, "value"])


def _render_structured_chart(chart_data: dict) -> bool:
    if not isinstance(chart_data, dict):
        return False

    chart_type = str(chart_data.get("chart_type") or "").lower()
    if chart_type == "forecast_multi_axis":
        return _render_forecast_multi_axis_chart(chart_data)

    # New generic contract path: x_axis/y_axis/series/data
    if isinstance(chart_data.get("x_axis"), dict) and isinstance(chart_data.get("data"), list) and isinstance(chart_data.get("series"), list):
        rendered = _render_new_contract_chart(chart_data)
        if rendered:
            return True

    series_list = chart_data.get("series")
    if not isinstance(series_list, list) or not series_list:
        return False

    chart_type = str(chart_data.get("chart_type") or "line").lower()
    x_field = str(chart_data.get("x_field") or "date").lower()
    if x_field not in {"date", "category"}:
        x_field = "date"
    fig = go.Figure()
    rendered_traces = 0

    for series in series_list:
        if not isinstance(series, dict):
            continue
        data_obj = series.get("data")
        if data_obj is None:
            continue

        df = pd.DataFrame(data_obj)
        df = _normalize_series_data(df, x_field)
        if df.empty:
            continue

        name = str(series.get("name") or series.get("kind") or f"Series {rendered_traces + 1}")
        kind = str(series.get("kind") or "forecast").lower()
        line_style = str(series.get("line_style") or ("dot" if kind == "forecast" else "solid"))
        line_color = "#111111" if kind == "actual" else "#b22222"
        marker_color = line_color

        if chart_type == "bar":
            fig.add_trace(
                go.Bar(
                    x=df[x_field],
                    y=df["value"],
                    name=name,
                    marker={"color": line_color},
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[x_field],
                    y=df["value"],
                    mode="lines+markers",
                    name=name,
                    line={"dash": line_style, "color": line_color},
                    marker={"color": marker_color},
                )
            )
        rendered_traces += 1

    if rendered_traces == 0:
        return False

    y_title = str(chart_data.get("measure") or chart_data.get("y_field") or "Value")
    title = str(chart_data.get("title") or "Forecast Trend")
    fig.update_layout(
        height=520,
        title=title,
        xaxis_title="Date" if x_field == "date" else "Category",
        yaxis_title=y_title,
        legend={"orientation": "h", "y": -0.2},
        barmode=str(chart_data.get("bar_mode") or "group") if chart_type == "bar" else None,
    )

    st.subheader("📈 Forecast Visualization")
    st.plotly_chart(fig, width="stretch")
    return True


def _render_new_contract_chart(chart_data: dict) -> bool:
    chart_type = str(chart_data.get("chart_type") or "line").lower()
    if chart_type not in {"line", "bar", "stacked_bar", "forecast_band"}:
        return False

    x_axis = chart_data.get("x_axis") or {}
    y_axis = chart_data.get("y_axis") or {}
    data_obj = chart_data.get("data")
    series = chart_data.get("series") or []
    if not isinstance(data_obj, list) or not isinstance(series, list):
        return False

    df = pd.DataFrame(data_obj)
    if df.empty:
        return False

    x_field = str(x_axis.get("field") or "")
    if not x_field or x_field not in df.columns:
        return False

    if str(x_axis.get("type") or "").lower() == "date":
        cleaned = _clean_x_labels(df[x_field])
        parsed = _parse_dates_with_known_formats(cleaned)
        if parsed is None:
            return False
        df[x_field] = parsed
        df = df.dropna(subset=[x_field]).sort_values(x_field)
    else:
        df[x_field] = _clean_x_labels(df[x_field])
        df = df[df[x_field].astype(str).str.strip() != ""]

    fig = go.Figure()
    traces = 0

    if chart_type == "forecast_band":
        if not isinstance(series, dict):
            return False
        fc = (series.get("forecast") or {}).get("field")
        ci = series.get("confidence_interval") or {}
        lo = ci.get("lower_field")
        hi = ci.get("upper_field")
        if not fc or not lo or not hi or fc not in df.columns or lo not in df.columns or hi not in df.columns:
            return False
        df[fc] = _to_numeric(df[fc])
        df[lo] = _to_numeric(df[lo])
        df[hi] = _to_numeric(df[hi])
        df = df.dropna(subset=[fc, lo, hi])
        if df.empty:
            return False

        fig.add_trace(go.Scatter(x=df[x_field], y=df[hi], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df[x_field], y=df[lo], mode="lines", fill="tonexty", fillcolor="rgba(178,34,34,0.2)", line={"width": 0}, name=ci.get("label") or "Confidence Interval"))
        fig.add_trace(go.Scatter(x=df[x_field], y=df[fc], mode="lines+markers", name=(series.get("forecast") or {}).get("label") or "Forecast", line={"color": "#b22222"}))
        traces = 3
    else:
        for item in series:
            if not isinstance(item, dict):
                continue
            field = item.get("field")
            label = str(item.get("label") or field or "Series")
            if not field or field not in df.columns:
                continue
            df[field] = _to_numeric(df[field])
            local = df.dropna(subset=[field])
            if local.empty:
                continue
            if chart_type in {"bar", "stacked_bar"}:
                fig.add_trace(go.Bar(x=local[x_field], y=local[field], name=label))
            else:
                fig.add_trace(go.Scatter(x=local[x_field], y=local[field], mode="lines+markers", name=label))
            traces += 1

    if traces == 0:
        return False

    title = str(chart_data.get("title") or "Forecast Trend")
    y_label = str(y_axis.get("label") or y_axis.get("field") or "Value")
    fig.update_layout(
        height=520,
        title=title,
        xaxis_title=str(x_axis.get("label") or x_field),
        yaxis_title=y_label,
        legend={"orientation": "h", "y": -0.2},
        barmode="stack" if chart_type == "stacked_bar" else ("group" if chart_type == "bar" else None),
    )

    st.subheader("📈 Forecast Visualization")
    st.plotly_chart(fig, width="stretch")
    return True


def _render_forecast_multi_axis_chart(chart_data: dict) -> bool:
    x_axis = chart_data.get("x_axis") or {}
    primary = chart_data.get("primary_y_axis") or {}
    secondary = chart_data.get("secondary_y_axis") or {}
    series = chart_data.get("series") or []
    data_obj = chart_data.get("data")

    if not isinstance(x_axis, dict) or not isinstance(primary, dict) or not isinstance(secondary, dict):
        return False
    if not isinstance(series, list) or not isinstance(data_obj, list):
        return False

    x_field = str(x_axis.get("field") or "")
    p_field = str(primary.get("field") or "")
    if not x_field or not p_field:
        return False

    s_defs = {str(item.get("type") or ""): item for item in series if isinstance(item, dict)}
    lo_field = str((s_defs.get("confidence_lower") or {}).get("field") or "")
    hi_field = str((s_defs.get("confidence_upper") or {}).get("field") or "")
    conf = float(chart_data.get("confidence_level") or 0.0)

    df = pd.DataFrame(data_obj)
    if df.empty or x_field not in df.columns or p_field not in df.columns:
        return False

    has_ci_fields = bool(lo_field and hi_field and lo_field in df.columns and hi_field in df.columns)

    cleaned_dates = _clean_x_labels(df[x_field])
    parsed = _parse_dates_with_known_formats(cleaned_dates)
    if parsed is None:
        return False
    df[x_field] = parsed

    df[p_field] = _to_numeric(df[p_field])
    req = [x_field, p_field]
    if has_ci_fields:
        df[lo_field] = _to_numeric(df[lo_field])
        req.append(lo_field)
        df[hi_field] = _to_numeric(df[hi_field])
        req.append(hi_field)

    df = df.dropna(subset=req).sort_values(x_field)
    if df.empty:
        return False

    show_ci = has_ci_fields
    if show_ci:
        valid = (df[lo_field] <= df[p_field]) & (df[p_field] <= df[hi_field])
        # If CI is malformed for many rows, degrade gracefully to predicted-only chart.
        if valid.sum() < max(2, int(len(df) * 0.5)):
            show_ci = False
        else:
            df = df[valid]
            if df.empty:
                show_ci = False

    group_field = str(chart_data.get("group_field") or "")
    if group_field and group_field not in df.columns:
        group_field = ""

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    groups = [None] if not group_field else [g for g in df[group_field].dropna().astype(str).unique().tolist()]
    if not groups:
        groups = [None]

    for g in groups:
        gdf = df if g is None else df[df[group_field].astype(str) == g]
        if gdf.empty:
            continue
        suffix = "" if g is None else f" ({g})"

        fig.add_trace(
            go.Scatter(
                x=gdf[x_field],
                y=gdf[p_field],
                mode="lines+markers",
                name=f"Predicted Demand{suffix}",
                line={"color": "#1f77b4", "dash": "solid"},
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Predicted=%{y:.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

        if show_ci:
            ci_label = f"{int(conf * 100)}% CI" if conf > 0 else "CI"
            fig.add_trace(
                go.Scatter(
                    x=gdf[x_field],
                    y=gdf[lo_field],
                    mode="lines",
                    name=f"Lower {ci_label}{suffix}",
                    line={"color": "#b22222", "dash": "dot"},
                    hovertemplate="Date=%{x|%Y-%m-%d}<br>Lower CI=%{y:.2f}<extra></extra>",
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=gdf[x_field],
                    y=gdf[hi_field],
                    mode="lines",
                    name=f"Upper {ci_label}{suffix}",
                    line={"color": "#8b0000", "dash": "dot"},
                    hovertemplate="Date=%{x|%Y-%m-%d}<br>Upper CI=%{y:.2f}<extra></extra>",
                ),
                secondary_y=True,
            )

    if not fig.data:
        return False

    fig.update_layout(
        height=520,
        title=str(chart_data.get("title") or "Demand Forecast"),
        legend={"orientation": "h", "y": -0.2},
        hovermode="x unified",
    )
    fig.update_xaxes(title_text=str(x_axis.get("label") or "Date"))
    fig.update_yaxes(title_text=str(primary.get("label") or "Predicted Demand (Units)"), secondary_y=False)
    fig.update_yaxes(
        title_text=str(secondary.get("label") or "Confidence Interval (Units)") if show_ci else "",
        secondary_y=True,
        showticklabels=show_ci,
    )

    st.subheader("📈 Forecast Visualization")
    st.plotly_chart(fig, width="stretch")
    return True


def _is_weekly_granularity(df: pd.DataFrame, x_col: str) -> bool:
    col_name = str(x_col).strip().lower()
    if "week" in col_name:
        return True

    series = df[x_col].astype(str).str.replace(r"\*", "", regex=True).str.strip().str.lower()
    if series.empty:
        return False

    week_like = series.str.contains(r"\bweek\b|\bweekly\b", regex=True, na=False)
    return week_like.mean() >= 0.5


def _filter_weekly_table_if_not_requested(df: pd.DataFrame, user_prompt: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    prompt_text = (user_prompt or "").lower()
    weekly_requested = bool(re.search(r"\bweek\b|\bweekly\b|\bweek-wise\b|\bweekwise\b", prompt_text))
    if weekly_requested:
        return df

    # Infer likely x-axis for granularity check.
    x_col = None
    columns = list(df.columns)
    for c in columns:
        lc = str(c).lower()
        if any(key in lc for key in ["date", "day", "month", "week"]):
            x_col = c
            break
    if x_col is None and columns:
        x_col = columns[0]

    if x_col is not None and _is_weekly_granularity(df, x_col):
        return None

    return df

if SIDEBAR_LOGO_PATH.exists():
    st.sidebar.image(str(SIDEBAR_LOGO_PATH), width="stretch")
st.sidebar.markdown("---")
st.sidebar.subheader("Agent Controls")

chat_url = CHAT_URL
refresh_agent_url = REFRESH_AGENT_URL

if st.sidebar.button("🔄 Refresh Agent"):
    with st.sidebar:
        with st.spinner("Refreshing agent..."):
            try:
                res = requests.post(refresh_agent_url, timeout=REQUEST_TIMEOUT_SECONDS)

                if res.status_code == 200:
                    st.success("✅ Agent refreshed successfully")
                else:
                    st.error(f"❌ Failed: {res.text}")

            except Exception as e:
                st.error(f"🚨 Error: {str(e)}")

# -------------------------------
# Session
# -------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:8]}"

thread_id = st.sidebar.text_input(
    "Thread ID",
    value=st.session_state.thread_id,
)
st.session_state.thread_id = thread_id

if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.rerun()


# -------------------------------
# Chat State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Title
# -------------------------------
st.title("Forecasting Agent")

# -------------------------------
# Display Chat History
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# User Input
# -------------------------------
if prompt := st.chat_input("Ask something..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                response = requests.post(
                    chat_url,
                    json={
                        "message": prompt,
                        "session_id": st.session_state.thread_id
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

                if response.ok:
                    data = response.json() if response.content else {}
                    print(data)
                    agent_payload = data.get("agent_payload") if isinstance(data.get("agent_payload"), dict) else {}
                    reply = (
                        data.get("reply")
                        or data.get("response")
                        or data.get("answer")
                        or data.get("message")
                        or "No response received."
                    )
                    latency = data.get("latency_ms", None)
                    data_type = data.get("type") or agent_payload.get("type")
                    structured_data = data.get("data") or agent_payload.get("data")
                    chart_data = data.get("chart_data") or agent_payload.get("chart_data")
                    
                    # Show text response
                    st.markdown(reply)

                    chart_rendered = False
                    forecast_intent = _is_forecast_intent(prompt, data_type, reply)
                    labor_mode = _is_labor_context(prompt, data_type, reply)

                    # 0) Use backend structured chart contract first when available.
                    if chart_data:
                        try:
                            chart_rendered = _render_structured_chart(chart_data)
                        except Exception:
                            chart_rendered = False

                    # 1) Try structured data from backend first
                    if not chart_rendered and structured_data:
                        try:
                            candidate_df = pd.DataFrame(structured_data)
                            candidate_df = _filter_weekly_table_if_not_requested(candidate_df, prompt)
                            if candidate_df is None:
                                raise ValueError("Weekly data skipped because user did not ask for weekly view.")
                            if labor_mode:
                                fig, chart_df, _ = _build_labor_daily_process_chart(candidate_df)
                                if fig is None and forecast_intent:
                                    fig, chart_df, _ = _build_actual_forecast_date_chart(candidate_df)
                                elif fig is None:
                                    fig, chart_df, _ = _build_line_chart(candidate_df)
                            elif forecast_intent:
                                fig, chart_df, _ = _build_actual_forecast_date_chart(candidate_df)
                                if fig is None:
                                    fig, chart_df, _ = _build_line_chart(candidate_df)
                            else:
                                fig, chart_df, _ = _build_line_chart(candidate_df)
                            if fig is None and labor_mode:
                                fig, chart_df, _ = _build_labor_table_chart(candidate_df)
                            if fig is not None:
                                st.subheader("📈 Forecast Visualization")
                                st.plotly_chart(fig, width="stretch")
                                with st.expander("📄 View Data"):
                                    st.dataframe(chart_df if chart_df is not None else candidate_df)
                                chart_rendered = True
                        except Exception:
                            chart_rendered = False

                    # 2) If no structured chart, parse markdown table(s) from model reply
                    if not chart_rendered:
                        markdown_tables = _extract_markdown_tables(reply)
                        for table_df in markdown_tables:
                            table_df = _filter_weekly_table_if_not_requested(table_df, prompt)
                            if table_df is None:
                                continue
                            if labor_mode:
                                fig, chart_df, _ = _build_labor_daily_process_chart(table_df)
                                if fig is None and forecast_intent:
                                    fig, chart_df, _ = _build_actual_forecast_date_chart(table_df)
                                elif fig is None:
                                    fig, chart_df, _ = _build_line_chart(table_df)
                            elif forecast_intent:
                                fig, chart_df, _ = _build_actual_forecast_date_chart(table_df)
                                if fig is None:
                                    fig, chart_df, _ = _build_line_chart(table_df)
                            else:
                                fig, chart_df, _ = _build_line_chart(table_df)
                            if fig is None and labor_mode:
                                fig, chart_df, _ = _build_labor_table_chart(table_df)
                            if fig is not None:
                                st.subheader("📈 Forecast Visualization")
                                st.plotly_chart(fig, width="stretch")
                                with st.expander("📄 View Parsed Table"):
                                    st.dataframe(chart_df if chart_df is not None else table_df)
                                chart_rendered = True
                                break

                    if not chart_rendered:
                        if forecast_intent:
                            if labor_mode:
                                st.caption("Forecast chart hidden: labor response did not include plottable category metrics or valid structured chart_data.")
                            else:
                                st.caption("Forecast chart hidden: need structured chart_data or a date-based table with clear Actual and Forecast columns.")
                        else:
                            st.caption("No date-level rows found for plotting. Showing textual forecast only.")

                    # latency
                    if latency:
                        st.caption(f"⏱ {round(latency, 2)} ms")

                    # save chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply
                    })

                else:
                    error_msg, ray_id = _format_backend_error(response)
                    st.error(error_msg)
                    if ray_id:
                        st.caption(f"Cloudflare Ray ID: {ray_id}")

            except Exception as e:
                st.error(f"Request failed: {str(e)}")
                
