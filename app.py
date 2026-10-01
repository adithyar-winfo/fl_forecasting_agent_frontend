import streamlit as st
import requests
import uuid
import pandas as pd
import plotly.graph_objects as go
import re
import os
from io import StringIO
from pathlib import Path

# -------------------------------
# Config
# -------------------------------
CHAT_URL = os.getenv("CHAT_URL", "https://fl-forecasting-agent-backend.onrender.com/agent/chat")
REFRESH_AGENT_URL = os.getenv("REFRESH_AGENT_URL", "https://fl-forecasting-agent-backend.onrender.com/agent/refresh_agent")
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


def _is_labor_context(prompt: str, data_type: str | None, reply: str) -> bool:
    text = f"{prompt} {data_type or ''} {reply}".lower()
    return "labor" in text or "labour" in text or "headcount" in text or "staffing" in text


def _normalize_series_data(series_data: pd.DataFrame) -> pd.DataFrame:
    if series_data.empty or "date" not in series_data.columns or "value" not in series_data.columns:
        return pd.DataFrame(columns=["date", "value"])

    working = series_data.copy()
    working["date"] = _clean_x_labels(working["date"])
    parsed = _parse_dates_with_known_formats(working["date"])
    if parsed is None:
        return pd.DataFrame(columns=["date", "value"])

    working["date"] = parsed
    working["value"] = _to_numeric(working["value"])
    working = working.dropna(subset=["date", "value"]).sort_values("date")
    return working[["date", "value"]]


def _render_structured_chart(chart_data: dict) -> bool:
    if not isinstance(chart_data, dict):
        return False

    series_list = chart_data.get("series")
    if not isinstance(series_list, list) or not series_list:
        return False

    chart_type = str(chart_data.get("chart_type") or "line").lower()
    fig = go.Figure()
    rendered_traces = 0

    for series in series_list:
        if not isinstance(series, dict):
            continue
        data_obj = series.get("data")
        if data_obj is None:
            continue

        df = pd.DataFrame(data_obj)
        df = _normalize_series_data(df)
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
                    x=df["date"],
                    y=df["value"],
                    name=name,
                    marker={"color": line_color},
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
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
        xaxis_title="Date",
        yaxis_title=y_title,
        legend={"orientation": "h", "y": -0.2},
        barmode=str(chart_data.get("bar_mode") or "group") if chart_type == "bar" else None,
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

if st.sidebar.button("🔄 Refresh Agent"):
    with st.sidebar:
        with st.spinner("Refreshing agent..."):
            try:
                res = requests.post(REFRESH_AGENT_URL, timeout=REQUEST_TIMEOUT_SECONDS)

                if res.status_code == 200:
                    st.success("✅ Agent refreshed successfully")
                else:
                    st.error(f"❌ Failed: {res.text}")

            except Exception as e:
                st.error(f"🚨 Error: {str(e)}")

# -------------------------------
# Sidebar
# -------------------------------
# Thread ID (conversation memory)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:8]}"

thread_id = st.sidebar.text_input(
    "Thread ID",
    value=st.session_state.thread_id
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
                    CHAT_URL,
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
                            fig, chart_df, _ = _build_line_chart(candidate_df)
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
                        labor_mode = _is_labor_context(prompt, data_type, reply)
                        for table_df in markdown_tables:
                            table_df = _filter_weekly_table_if_not_requested(table_df, prompt)
                            if table_df is None:
                                continue
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