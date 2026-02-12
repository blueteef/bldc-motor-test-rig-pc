import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BLDC Run Compare", layout="wide")


def load_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_csv(index_path)
    if "run_id" in df.columns:
        df["run_id"] = df["run_id"].astype(str).str.zfill(4)
    return df


def read_run_csv(csv_path: str) -> pd.DataFrame:
    rows = []
    header = None
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            header = [h.strip() for h in line.strip().split(",")]
            break
        if header is None:
            return pd.DataFrame()
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            rows.append(parts[: len(header)])

    df = pd.DataFrame(rows, columns=header)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def prep_run_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp_ms" in df.columns:
        t = df["timestamp_ms"].to_numpy(dtype=float)
        t0 = np.nanmin(t) if np.isfinite(t).any() else 0.0
        df["t_s"] = (df["timestamp_ms"] - t0) / 1000.0
    else:
        df["t_s"] = np.arange(len(df), dtype=float)

    if "rpm" in df.columns:
        df.loc[df["rpm"] < 0, "rpm"] = np.nan
    return df


def overlay_time_plot(df_a, df_b, y_col: str, name_a: str, name_b: str, title: str, y_title: str | None = None) -> go.Figure:
    fig = go.Figure()
    if y_col in df_a.columns:
        fig.add_trace(go.Scatter(x=df_a["t_s"], y=df_a[y_col], mode="lines", name=name_a))
    if y_col in df_b.columns:
        fig.add_trace(go.Scatter(x=df_b["t_s"], y=df_b[y_col], mode="lines", name=name_b))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_title or y_col,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
    )
    return fig


def scatter_plot(df_a, df_b, x_col: str, y_col: str, name_a: str, name_b: str, title: str) -> go.Figure:
    fig = go.Figure()
    if x_col in df_a.columns and y_col in df_a.columns:
        fig.add_trace(go.Scatter(x=df_a[x_col], y=df_a[y_col], mode="markers", name=name_a, opacity=0.55))
    if x_col in df_b.columns and y_col in df_b.columns:
        fig.add_trace(go.Scatter(x=df_b[x_col], y=df_b[y_col], mode="markers", name=name_b, opacity=0.55))
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
    )
    return fig


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


st.title("BLDC Run Compare")

with st.sidebar:
    st.header("Paths")
    index_path_txt = st.text_input("Index file", value="bldc_loader/data/runs_index.csv")
    runs_dir_txt = st.text_input("Runs folder filter (optional; prefix match)", value="")
    st.divider()
    st.caption("Debug")
    st.write("CWD:", os.getcwd())

ip = Path(index_path_txt).expanduser()
if not ip.exists():
    st.error(f"Index file not found: {ip.resolve()}")
    st.stop()

dfi = load_index(ip)

if runs_dir_txt.strip():
    rd = str(Path(runs_dir_txt).expanduser().resolve())
    dfi = dfi[dfi["csv_path"].astype(str).str.startswith(rd)]

st.subheader("Run library")

col1, col2, col3 = st.columns([2, 2, 4])
with col1:
    motors = ["(all)"] + sorted([m for m in dfi["motor_id"].fillna("").unique().tolist() if str(m).strip()])
    motor_filter = st.selectbox("Filter: motor_id", motors, index=0)
with col2:
    fws = ["(all)"] + sorted([m for m in dfi["fw"].fillna("").unique().tolist() if str(m).strip()])
    fw_filter = st.selectbox("Filter: fw", fws, index=0)
with col3:
    show_cols = st.multiselect(
        "Columns to show",
        [
            "run_id",
            "run_uid",
            "motor_id",
            "fw",
            "duration_s",
            "kv_median",
            "v_mean",
            "i_mean",
            "vib_rms_mean",
            "vib_rms_max",
            "rpm_max",
            "dt_std_ms",
            "json_status",
        ],
        default=[
            "run_id",
            "run_uid",
            "motor_id",
            "fw",
            "duration_s",
            "kv_median",
            "v_mean",
            "i_mean",
            "vib_rms_mean",
            "vib_rms_max",
            "rpm_max",
            "dt_std_ms",
            "json_status",
        ],
    )

dff = dfi.copy()
if motor_filter != "(all)":
    dff = dff[dff["motor_id"] == motor_filter]
if fw_filter != "(all)":
    dff = dff[dff["fw"] == fw_filter]

st.dataframe(dff[show_cols].sort_values("run_id"), use_container_width=True, height=240)

st.divider()
st.subheader("Compare two runs")

run_ids = dff["run_id"].astype(str).str.zfill(4).tolist()
if len(run_ids) < 2:
    st.warning("Need at least 2 runs in the current filtered set to compare.")
    st.stop()

a_id = st.selectbox("Run A", run_ids, index=max(0, len(run_ids) - 2))
b_id = st.selectbox("Run B", run_ids, index=max(0, len(run_ids) - 1))

row_a = dfi[dfi["run_id"] == a_id].iloc[0]
row_b = dfi[dfi["run_id"] == b_id].iloc[0]

name_a = f"RUN{a_id} ({row_a.get('motor_id','')})"
name_b = f"RUN{b_id} ({row_b.get('motor_id','')})"

# Prominent identifiers + metrics
st.markdown("### Key identifiers")
id1, id2 = st.columns(2)
with id1:
    st.code(f"RUN{a_id}  run_uid={row_a.get('run_uid','')}", language="text")
with id2:
    st.code(f"RUN{b_id}  run_uid={row_b.get('run_uid','')}", language="text")

st.markdown("### Key metrics")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("kV (A)", fmt(row_a.get("kv_median"), 0), help=f"Median(rpm/volts). p10–p90: {fmt(row_a.get('kv_p10'),0)}–{fmt(row_a.get('kv_p90'),0)}")
m2.metric("kV (B)", fmt(row_b.get("kv_median"), 0), help=f"Median(rpm/volts). p10–p90: {fmt(row_b.get('kv_p10'),0)}–{fmt(row_b.get('kv_p90'),0)}")
m3.metric("Volts mean (A)", fmt(row_a.get("v_mean"), 2))
m4.metric("Volts mean (B)", fmt(row_b.get("v_mean"), 2))
m5.metric("Amps mean (A)", fmt(row_a.get("i_mean"), 2))
m6.metric("Amps mean (B)", fmt(row_b.get("i_mean"), 2))

st.caption(
    f"Voltage column: A={row_a.get('volt_col','') or '—'} | B={row_b.get('volt_col','') or '—'}    •    "
    f"Current column: A={row_a.get('curr_col','') or '—'} | B={row_b.get('curr_col','') or '—'}"
)

# Load full data
csv_a = str(row_a["csv_path"])
csv_b = str(row_b["csv_path"])
df_a = prep_run_df(read_run_csv(csv_a))
df_b = prep_run_df(read_run_csv(csv_b))

# Summary table
st.markdown("### Summary (A vs B)")
summary_keys = [
    "run_uid", "csv_sha256",
    "motor_id", "fw", "git_sha", "build_utc",
    "n_rows", "duration_s",
    "log_hz_declared", "fs_hz_est", "dt_med_ms", "dt_std_ms",
    "kv_median", "kv_p10", "kv_p90",
    "v_mean", "v_min", "v_max",
    "i_mean", "i_max",
    "vib_rms_mean", "vib_rms_max",
    "rpm_mean", "rpm_max",
    "json_status",
]
summary = pd.DataFrame([{"metric": k, "A": row_a.get(k, ""), "B": row_b.get(k, "")} for k in summary_keys])
st.dataframe(summary, use_container_width=True, height=460)

# Plots
st.markdown("### Plots")
p1, p2 = st.columns(2)
with p1:
    st.plotly_chart(
        overlay_time_plot(df_a, df_b, "vib_rms_mps2", name_a, name_b, "Vibration RMS vs Time"),
        use_container_width=True,
    )
with p2:
    st.plotly_chart(
        overlay_time_plot(df_a, df_b, "rpm", name_a, name_b, "RPM vs Time"),
        use_container_width=True,
    )

p3, p4 = st.columns(2)
with p3:
    st.plotly_chart(
        scatter_plot(df_a, df_b, "rpm", "vib_rms_mps2", name_a, name_b, "Vibration RMS vs RPM"),
        use_container_width=True,
    )
with p4:
    st.plotly_chart(
        scatter_plot(df_a, df_b, "thr_cmd", "vib_rms_mps2", name_a, name_b, "Vibration RMS vs Throttle Command"),
        use_container_width=True,
    )

# Voltage/current plots (supports differing column names between runs)
vcol_a = str(row_a.get("volt_col", "") or "")
vcol_b = str(row_b.get("volt_col", "") or "")
icol_a = str(row_a.get("curr_col", "") or "")
icol_b = str(row_b.get("curr_col", "") or "")

p5, p6 = st.columns(2)
with p5:
    fig = go.Figure()
    if vcol_a and vcol_a in df_a.columns:
        fig.add_trace(go.Scatter(x=df_a["t_s"], y=df_a[vcol_a], mode="lines", name=f"{name_a} ({vcol_a})"))
    if vcol_b and vcol_b in df_b.columns:
        fig.add_trace(go.Scatter(x=df_b["t_s"], y=df_b[vcol_b], mode="lines", name=f"{name_b} ({vcol_b})"))
    if len(fig.data) == 0:
        st.info("No voltage column found in either run (configure VOLT_COL_CANDIDATES in index_runs.py).")
    else:
        fig.update_layout(title="Voltage vs Time", xaxis_title="Time (s)", yaxis_title="Volts", height=320,
                          margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

with p6:
    fig = go.Figure()
    if icol_a and icol_a in df_a.columns:
        fig.add_trace(go.Scatter(x=df_a["t_s"], y=df_a[icol_a], mode="lines", name=f"{name_a} ({icol_a})"))
    if icol_b and icol_b in df_b.columns:
        fig.add_trace(go.Scatter(x=df_b["t_s"], y=df_b[icol_b], mode="lines", name=f"{name_b} ({icol_b})"))
    if len(fig.data) == 0:
        st.info("No current column found in either run (configure CURR_COL_CANDIDATES in index_runs.py).")
    else:
        fig.update_layout(title="Current vs Time", xaxis_title="Time (s)", yaxis_title="Amps", height=320,
                          margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
