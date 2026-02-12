#!/usr/bin/env python3
"""
Build a single-file run index from a folder containing RUN####.CSV (+ optional RUN####.JSON).

Assumptions:
- runs are in ONE folder (non-recursive)
- CSV is source of truth
- CSV has "# key=value" metadata header lines
- Data header row then numeric rows

Adds:
- run_uid derived from SHA256(csv bytes) (first 16 chars for display)
- csv_sha256 full digest
- voltage/current summary fields (vin_v / iin_a supported)
- kV estimates geared for NO-LOAD (factory-ish):
    * kv_top_*: median(rpm/v) on the top-RPM slice (closest to steady no-load)
    * kv_fit  : regression-corrected using V ≈ a*RPM + b*I + c => kV=1/a (cross-check)

Usage:
  python index_runs.py --runs-dir "C:/path/to/SD_dump" --out "bldc_loader/data/runs_index.csv"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

RUN_RE = re.compile(r"^RUN(\d{4})\.CSV$", re.IGNORECASE)

VOLT_COL_CANDIDATES = ["vin_v", "vbat_v", "vbatt_v", "vbus_v", "volts", "voltage_v", "pack_v"]
CURR_COL_CANDIDATES = ["iin_a", "ibat_a", "current_a", "amps", "current"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_kv_header(csv_path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                break
            s = s.lstrip("#").strip()
            if "=" in s:
                k, v = s.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def repair_json_text(txt: str) -> str:
    cleaned = txt.strip()
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    opens = cleaned.count("{")
    closes = cleaned.count("}")
    if opens > closes:
        cleaned += "}" * (opens - closes)

    opens = cleaned.count("[")
    closes = cleaned.count("]")
    if opens > closes:
        cleaned += "]" * (opens - closes)

    return cleaned


def try_load_json(json_path: Path) -> Tuple[str, Dict[str, Any]]:
    if not json_path.exists():
        return "none", {}
    txt = json_path.read_text(encoding="utf-8", errors="replace")
    if not txt.strip():
        return "invalid", {}
    try:
        return "ok", json.loads(txt)
    except Exception:
        try:
            return "repaired", json.loads(repair_json_text(txt))
        except Exception:
            return "invalid", {}


def read_csv_data(csv_path: Path) -> pd.DataFrame:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        lines = (ln for ln in f if not ln.lstrip().startswith("#"))
        reader = csv.reader(lines)
        header = next(reader, None)
        if header is None:
            return pd.DataFrame()
        rows = list(reader)
        df = pd.DataFrame(rows, columns=[h.strip() for h in header])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def compute_dt_stats_ms(ts_ms: np.ndarray) -> Dict[str, Optional[float]]:
    ts = ts_ms.astype(float)
    ts = ts[np.isfinite(ts)]
    if ts.size < 3:
        return dict(dt_med_ms=None, dt_std_ms=None, dt_p10_ms=None, dt_p90_ms=None, fs_hz_est=None)

    d = np.diff(ts)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return dict(dt_med_ms=None, dt_std_ms=None, dt_p10_ms=None, dt_p90_ms=None, fs_hz_est=None)

    dt_med = float(np.median(d))
    dt_std = float(np.std(d, ddof=1)) if d.size >= 3 else float(np.std(d))
    dt_p10 = float(np.percentile(d, 10))
    dt_p90 = float(np.percentile(d, 90))
    fs = float(1000.0 / dt_med) if dt_med > 0 else None

    return dict(dt_med_ms=dt_med, dt_std_ms=dt_std, dt_p10_ms=dt_p10, dt_p90_ms=dt_p90, fs_hz_est=fs)


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_voltage_current_summary(df: pd.DataFrame) -> Dict[str, Optional[float] | str]:
    out: Dict[str, Optional[float] | str] = {
        "volt_col": "",
        "curr_col": "",
        "v_mean": None,
        "v_min": None,
        "v_max": None,
        "i_mean": None,
        "i_max": None,
    }

    vcol = pick_first_existing(df, VOLT_COL_CANDIDATES)
    icol = pick_first_existing(df, CURR_COL_CANDIDATES)

    if vcol:
        out["volt_col"] = vcol
        v = df[vcol].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            out["v_mean"] = float(np.mean(v))
            out["v_min"] = float(np.min(v))
            out["v_max"] = float(np.max(v))

    if icol:
        out["curr_col"] = icol
        i = df[icol].to_numpy(dtype=float)
        i = i[np.isfinite(i)]
        if i.size:
            out["i_mean"] = float(np.mean(i))
            out["i_max"] = float(np.max(i))

    return out


def compute_kv_estimates_no_load(df: pd.DataFrame, vcol: str, icol: str) -> Dict[str, Optional[float]]:
    """
    NO-LOAD / factory-ish estimates.

    kv_top_*:
      Use the top-RPM slice (top 20% of RPM values, after RPM>3000) and compute rpm/v.
      This approximates stabilized no-load RPM/Volt.

    kv_fit:
      Regression cross-check using V ≈ a*RPM + b*I + c => kV=1/a

    Returns keys:
      kv_top_median, kv_top_p10, kv_top_p90
      kv_fit, r_fit_ohm, v0_fit_v, kv_fit_r2
    """
    out: Dict[str, Optional[float]] = {
        "kv_top_median": None,
        "kv_top_p10": None,
        "kv_top_p90": None,
        "kv_fit": None,
        "r_fit_ohm": None,
        "v0_fit_v": None,
        "kv_fit_r2": None,
    }

    if "rpm" not in df.columns or vcol not in df.columns:
        return out

    rpm = df["rpm"].to_numpy(dtype=float)
    v = df[vcol].to_numpy(dtype=float)

    have_i = bool(icol) and (icol in df.columns)
    i = df[icol].to_numpy(dtype=float) if have_i else None

    m = np.isfinite(rpm) & np.isfinite(v) & (rpm > 0) & (v > 1.0)
    if have_i:
        m = m & np.isfinite(i)

    if not np.any(m):
        return out

    rpm_m = rpm[m]
    v_m = v[m]
    i_m = i[m] if have_i else None

    # drop invalid tach sentinel and startup region
    m2 = rpm_m >= 3000.0
    rpm_m = rpm_m[m2]
    v_m = v_m[m2]
    i_m = i_m[m2] if have_i else None

    if rpm_m.size < 30:
        return out

    # --- kv_top_*: top 20% RPM slice ---
    r_thr = float(np.percentile(rpm_m, 80.0))
    mt = rpm_m >= r_thr
    if np.count_nonzero(mt) >= 10:
        kv = rpm_m[mt] / v_m[mt]
        kv = kv[np.isfinite(kv)]
        if kv.size:
            out["kv_top_median"] = float(np.median(kv))
            out["kv_top_p10"] = float(np.percentile(kv, 10))
            out["kv_top_p90"] = float(np.percentile(kv, 90))

    # --- kv_fit: regression correction (cross-check) ---
    if have_i and i_m is not None and rpm_m.size >= 50:
        # avoid rare spikes dominating fit
        i_hi = float(np.percentile(i_m, 95.0))
        mf = i_m <= i_hi

        rpm_f = rpm_m[mf]
        v_f = v_m[mf]
        i_f = i_m[mf]

        if rpm_f.size >= 30:
            X = np.column_stack([rpm_f, i_f, np.ones_like(rpm_f)])
            y = v_f
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b, c = float(beta[0]), float(beta[1]), float(beta[2])

            if a > 0:
                out["kv_fit"] = float(1.0 / a)
                out["r_fit_ohm"] = float(b)
                out["v0_fit_v"] = float(c)

                yhat = X @ beta
                ss_res = float(np.sum((y - yhat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                out["kv_fit_r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

    return out


def build_index_for_run(csv_path: Path) -> Dict[str, Any]:
    m = RUN_RE.match(csv_path.name)
    if not m:
        raise ValueError(f"Not a run CSV: {csv_path}")
    run_id = m.group(1)

    json_path = csv_path.with_suffix(".JSON")

    csv_sha256 = sha256_file(csv_path)
    run_uid = csv_sha256[:16]

    meta = parse_kv_header(csv_path)
    json_status, _ = try_load_json(json_path)

    df = read_csv_data(csv_path)

    ts_col = "timestamp_ms"
    vib_col = "vib_rms_mps2"
    rpm_col = "rpm"
    thr_col = "thr_cmd"

    n_rows = int(df.shape[0])
    cols = list(df.columns)

    duration_ms = None
    if ts_col in df.columns:
        ts = df[ts_col].to_numpy(dtype=float)
        ts_f = ts[np.isfinite(ts)]
        if ts_f.size >= 2:
            duration_ms = float(ts_f[-1] - ts_f[0])

    timing = compute_dt_stats_ms(df[ts_col].to_numpy(dtype=float)) if ts_col in df.columns else {
        "dt_med_ms": None, "dt_std_ms": None, "dt_p10_ms": None, "dt_p90_ms": None, "fs_hz_est": None
    }

    vib_mean = vib_max = None
    if vib_col in df.columns:
        v = df[vib_col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            vib_mean = float(np.mean(v))
            vib_max = float(np.max(v))

    rpm_mean = rpm_max = None
    if rpm_col in df.columns:
        r = df[rpm_col].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        r = r[r >= 0]
        if r.size:
            rpm_mean = float(np.mean(r))
            rpm_max = float(np.max(r))

    thr_mean = thr_max = None
    if thr_col in df.columns:
        t = df[thr_col].to_numpy(dtype=float)
        t = t[np.isfinite(t)]
        if t.size:
            thr_mean = float(np.mean(t))
            thr_max = float(np.max(t))

    motor_id = meta.get("motor_id", "")
    fw = meta.get("fw", "")
    git_sha = meta.get("git_sha", "")
    build_utc = meta.get("build_utc", "")
    log_hz_declared = safe_float(meta.get("log_hz"))
    sample_hz = safe_float(meta.get("sample_hz"))

    vc = compute_voltage_current_summary(df)
    vcol = str(vc["volt_col"] or "")
    icol = str(vc["curr_col"] or "")

    kvs = compute_kv_estimates_no_load(df, vcol=vcol, icol=icol) if vcol else {
        "kv_top_median": None, "kv_top_p10": None, "kv_top_p90": None,
        "kv_fit": None, "r_fit_ohm": None, "v0_fit_v": None, "kv_fit_r2": None
    }

    # "factory-like" preferred kV
    kv_factory = kvs["kv_top_median"] if kvs["kv_top_median"] is not None else kvs["kv_fit"]

    return {
        "run_id": run_id,
        "run_uid": run_uid,
        "csv_sha256": csv_sha256,
        "csv_path": str(csv_path),
        "json_path": str(json_path) if json_path.exists() else "",
        "json_status": json_status,
        "fw": fw,
        "git_sha": git_sha,
        "build_utc": build_utc,
        "motor_id": motor_id,
        "n_rows": n_rows,
        "duration_ms": duration_ms,
        "duration_s": (duration_ms / 1000.0) if duration_ms is not None else None,
        "dt_med_ms": timing["dt_med_ms"],
        "dt_std_ms": timing["dt_std_ms"],
        "dt_p10_ms": timing["dt_p10_ms"],
        "dt_p90_ms": timing["dt_p90_ms"],
        "fs_hz_est": timing["fs_hz_est"],
        "log_hz_declared": log_hz_declared,
        "sample_hz": sample_hz,
        "vib_rms_mean": vib_mean,
        "vib_rms_max": vib_max,
        "rpm_mean": rpm_mean,
        "rpm_max": rpm_max,
        "thr_mean": thr_mean,
        "thr_max": thr_max,
        "volt_col": vcol,
        "curr_col": icol,
        "v_mean": vc["v_mean"],
        "v_min": vc["v_min"],
        "v_max": vc["v_max"],
        "i_mean": vc["i_mean"],
        "i_max": vc["i_max"],
        "kv_factory": kv_factory,
        "kv_top_median": kvs["kv_top_median"],
        "kv_top_p10": kvs["kv_top_p10"],
        "kv_top_p90": kvs["kv_top_p90"],
        "kv_fit": kvs["kv_fit"],
        "r_fit_ohm": kvs["r_fit_ohm"],
        "v0_fit_v": kvs["v0_fit_v"],
        "kv_fit_r2": kvs["kv_fit_r2"],
        "columns": ",".join(cols),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="Folder containing RUN####.CSV files (one folder).")
    ap.add_argument("--out", default="bldc_loader/data/runs_index.csv", help="Output CSV path.")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    if not runs_dir.exists():
        raise SystemExit(f"runs-dir not found: {runs_dir}")

    csvs = sorted([p for p in runs_dir.iterdir() if p.is_file() and RUN_RE.match(p.name)])
    if not csvs:
        raise SystemExit(f"No RUN####.CSV files found in: {runs_dir}")

    rows = [build_index_for_run(p) for p in csvs]
    df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote index: {out}  (runs={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
