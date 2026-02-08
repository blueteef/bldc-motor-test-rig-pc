from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re

import pandas as pd

from .models import Run, RunSet


# Columns expected from firmware contract (time-series)
EXPECTED_COLS = [
    "timestamp_ms",
    "ax_mps2", "ay_mps2", "az_mps2",
    "mag_mps2",
    "vib_inst_mps2",
    "vib_rms_mps2",
    "vib_net_mps2",
    "thr_cmd",
    "esc_us",
    "rpm",
]


@dataclass
class LoadOptions:
    # Basic normalization
    add_time_s: bool = True
    add_g_units: bool = True
    sort_by_time: bool = True

    # Keep unknown CSV columns by default (forward-compat)
    drop_unknown_cols: bool = False

    # Folder-level extras
    load_summaries: bool = True

    # Parquet caching (speed up reloads)
    write_parquet_cache: bool = True
    overwrite_parquet: bool = False
    prefer_parquet_cache: bool = True

    # Sampling rate detection
    detect_fs: bool = True
    fs_quantile: float = 0.5
    fs_min_samples: int = 20

    # CSV parsing robustness
    csv_comment_char: str = "#"
    csv_sep: str = ","
    csv_engine: str = "python"

    # JSON tolerance
    tolerate_bad_json: bool = True
    # If JSON is invalid even after repair, do we still load the run?
    # (Recommended True: CSV is primary.)
    allow_missing_or_invalid_json: bool = True


def _parse_run_id(stem: str) -> Optional[int]:
    s = stem.upper()
    if not s.startswith("RUN"):
        return None
    digits = s[3:]
    if not digits.isdigit():
        return None
    return int(digits)


def _parquet_path_for(csv_path: Path) -> Path:
    return csv_path.with_suffix(".parquet")


def _meta_cache_path_for(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def _read_run_csv(csv_path: Path, opt: LoadOptions) -> pd.DataFrame:
    # Ignore metadata lines starting with '#'
    return pd.read_csv(
        csv_path,
        comment=opt.csv_comment_char,
        sep=opt.csv_sep,
        engine=opt.csv_engine,
        skip_blank_lines=True,
    )


def _repair_json_text(text: str) -> str:
    """
    Best-effort JSON repair:
    - Remove trailing commas before } or ]
    - Remove a trailing comma at EOF
    This fixes the exact errors you're seeing ("Illegal trailing comma...").
    """
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r",\s*\Z", "", text)
    return text


def _read_json_sidecar(path: Path, tolerate: bool) -> Tuple[Dict, List[str]]:
    """
    Returns (meta_dict, issues).
    Never raises unless tolerate=False and JSON is invalid.
    """
    issues: List[str] = []
    raw = path.read_text(encoding="utf-8", errors="replace")

    # First try strict
    try:
        return json.loads(raw), issues
    except json.JSONDecodeError as e:
        issues.append(f"json_invalid:{e}")

    if not tolerate:
        # escalate
        raise

    # Try repaired
    repaired = _repair_json_text(raw)
    try:
        meta = json.loads(repaired)
        issues.append("json_repaired_trailing_commas")
        return meta, issues
    except json.JSONDecodeError as e:
        issues.append(f"json_repair_failed:{e}")
        # Return empty meta; caller decides whether to allow
        return {}, issues


def _detect_fs_hz_from_timestamp_ms(df: pd.DataFrame, q: float, min_samples: int) -> Optional[float]:
    if "timestamp_ms" not in df.columns:
        return None

    ts = pd.to_numeric(df["timestamp_ms"], errors="coerce").dropna().values
    if len(ts) < (min_samples + 1):
        return None

    ts = pd.Series(ts).sort_values().values
    dt_ms = pd.Series(ts).diff().dropna()

    dt_ms = dt_ms[(dt_ms > 0) & (dt_ms < 10_000)]
    if len(dt_ms) < min_samples:
        return None

    dt_ms_q = float(dt_ms.quantile(q))
    if dt_ms_q <= 0:
        return None

    return 1000.0 / dt_ms_q


def _postprocess(df: pd.DataFrame, opt: LoadOptions, issues: List[str]) -> pd.DataFrame:
    if opt.drop_unknown_cols:
        keep = [c for c in df.columns if c in EXPECTED_COLS]
        df = df[keep].copy()

    numeric_cols = [
        "timestamp_ms", "thr_cmd", "esc_us", "rpm",
        "ax_mps2", "ay_mps2", "az_mps2", "mag_mps2",
        "vib_inst_mps2", "vib_rms_mps2", "vib_net_mps2",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if opt.add_time_s and "timestamp_ms" in df.columns:
        df["time_s"] = df["timestamp_ms"] / 1000.0

    if opt.sort_by_time and "timestamp_ms" in df.columns:
        df = df.sort_values("timestamp_ms").reset_index(drop=True)

    if opt.add_g_units:
        g = 9.80665
        for c in [
            "ax_mps2", "ay_mps2", "az_mps2", "mag_mps2",
            "vib_inst_mps2", "vib_rms_mps2", "vib_net_mps2",
        ]:
            if c in df.columns:
                df[c.replace("_mps2", "_g")] = df[c] / g

    if "rpm" not in df.columns or df["rpm"].isna().all():
        issues.append("rpm_missing_or_empty")

    return df


def _write_run_cache(run: Run, overwrite: bool) -> None:
    pq = _parquet_path_for(run.csv_path)
    mj = _meta_cache_path_for(run.csv_path)

    if pq.exists() and not overwrite:
        return

    run.df.to_parquet(pq, index=False)

    if run.meta:
        mj.write_text(json.dumps(run.meta, indent=2), encoding="utf-8")


def _read_summary_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    # Peek first non-empty line
    first = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                first = line
                break

    if first.startswith("run_id,"):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df

    # Short summary with no header (10 columns)
    short_cols = [
        "run_id",
        "start_ms",
        "end_ms",
        "duration_ms",
        "mode",
        "thr_peak",
        "esc_us_peak",
        "vib_max_g",
        "vib_rms_g",
        "rpm_max",
    ]
    return pd.read_csv(path, header=None, names=short_cols)


def load_run(csv_path: Path, opt: Optional[LoadOptions] = None) -> Run:
    opt = opt or LoadOptions()
    csv_path = Path(csv_path)

    run_id = _parse_run_id(csv_path.stem)
    if run_id is None:
        raise ValueError(f"Not a RUN#### CSV: {csv_path.name}")

    # Prefer parquet cache
    pq_path = _parquet_path_for(csv_path)
    if opt.prefer_parquet_cache and pq_path.exists():
        df = pd.read_parquet(pq_path)
    else:
        df = _read_run_csv(csv_path, opt)

    # Sidecar JSON (optional + tolerant)
    json_path = csv_path.with_suffix(".JSON")
    meta: Dict = {}
    issues: List[str] = []

    if json_path.exists():
        meta, json_issues = _read_json_sidecar(json_path, tolerate=opt.tolerate_bad_json)
        issues.extend(json_issues)
        if (not meta) and (not opt.allow_missing_or_invalid_json):
            raise ValueError(f"JSON invalid for {json_path.name}: {json_issues[-1] if json_issues else 'unknown'}")
    else:
        json_path = None
        issues.append("json_missing")
        if not opt.allow_missing_or_invalid_json:
            raise ValueError(f"JSON missing for {csv_path.name}")

    # Normalize dataframe
    df = _postprocess(df, opt, issues)

    # Derived sampling rate
    if opt.detect_fs:
        fs = _detect_fs_hz_from_timestamp_ms(df, q=opt.fs_quantile, min_samples=opt.fs_min_samples)
        if fs is not None:
            meta.setdefault("derived", {})
            meta["derived"]["fs_hz"] = fs
            meta["derived"]["dt_ms_est"] = 1000.0 / fs
        else:
            issues.append("fs_detect_failed")

    run = Run(
        run_id=run_id,
        csv_path=csv_path,
        json_path=json_path,
        df=df,
        meta=meta,
        issues=issues,
    )

    # Write caches
    if opt.write_parquet_cache:
        try:
            _write_run_cache(run, overwrite=opt.overwrite_parquet)
        except Exception as e:
            run.issues.append(f"parquet_cache_failed:{e}")

    return run


def load_folder(folder: Path, opt: Optional[LoadOptions] = None) -> RunSet:
    folder = Path(folder)
    opt = opt or LoadOptions()

    rs = RunSet(folder=folder)

    # Runs: only RUN####.CSV
    for csv_path in sorted(folder.glob("RUN[0-9][0-9][0-9][0-9].CSV")):
        try:
            rs.runs.append(load_run(csv_path, opt))
        except Exception as e:
            rs.issues.append(f"failed_load:{csv_path.name}:{e}")

    # Summaries
    if opt.load_summaries:
        rs.summary = _read_summary_csv(folder / "RUN_SUMMARY.CSV")
        if rs.summary is None:
            rs.issues.append("summary_missing:RUN_SUMMARY.CSV")

        rs.summary_short = _read_summary_csv(folder / "RUN_SUMMARY_SHORT.CSV")
        if rs.summary_short is None:
            rs.issues.append("summary_missing:RUN_SUMMARY_SHORT.CSV")

    return rs
