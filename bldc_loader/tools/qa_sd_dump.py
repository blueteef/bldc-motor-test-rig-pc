#!/usr/bin/env python3
"""
SD dump QA gatekeeper for BLDC motor test rig logs.

- Accepts only RUN####.CSV as "runs"
- CSV is source of truth
- JSON sidecar is optional + best-effort repair
- Produces qa_report.json + qa_report.csv

Usage:
  python qa_sd_dump.py /path/to/SD_dump_unzipped --out /path/to/out
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

# --------------------------
# Config (tune as needed)
# --------------------------
# Jitter thresholds (warn-only)
DT_STD_WARN_MS = 3.0        # stddev > 3ms => timing jitter notable
DT_MAX_WARN_MS = 40.0       # single-step outliers
DT_MIN_WARN_MS = 5.0        # too-fast steps (often catch-up artifacts)

RUN_CSV_RE = re.compile(r"^RUN(\d{4})\.CSV$", re.IGNORECASE)
RUN_JSON_FMT = "RUN{run_id}.JSON"

MIN_ROWS = 20

# Sampling bounds (seconds). Adjust to your known logger rate.
# Example for ~50 Hz => dt ~0.02s
DT_MED_MIN = 0.015
DT_MED_MAX = 0.025

# Core columns: if present, we validate; if missing, we flag.
CORE_COLS = ["timestamp_ms"]  # you can add "thr_cmd", "rpm", "vib_rms_mps2", etc.

# For NaN ratio checks, these are the columns that matter most if present.
QA_NUMERIC_COLS = ["timestamp_ms", "thr_cmd", "rpm", "vib_inst_mps2", "vib_rms_mps2"]

MAX_NAN_FRAC_CORE = 0.20  # 20%


# --------------------------
# Data structures
# --------------------------

@dataclass
class RunQA:
    dt_min_ms: Optional[float] = None
    dt_max_ms: Optional[float] = None
    dt_std_ms: Optional[float] = None
    dt_p10_ms: Optional[float] = None
    dt_p90_ms: Optional[float] = None

    run_id: str
    csv_path: str
    json_path: Optional[str] = None

    discovered: bool = True
    valid: bool = True
    flags: List[str] = field(default_factory=list)

    n_rows: int = 0
    columns: List[str] = field(default_factory=list)

    # timebase stats
    time_monotonic: Optional[bool] = None
    dt_median_s: Optional[float] = None
    fs_hz_est: Optional[float] = None

    # missing columns
    missing_core_cols: List[str] = field(default_factory=list)

    # NaN metrics
    nan_frac: Dict[str, float] = field(default_factory=dict)

    # JSON metadata status
    json_status: str = "none"   # none | ok | repaired | invalid
    json_meta: Dict = field(default_factory=dict)


@dataclass
class Report:
    root: str
    n_runs: int
    n_valid: int
    n_invalid: int
    fs_hz_hist: Dict[str, int]
    columns_union: List[str]
    runs: List[RunQA]


# --------------------------
# Helpers
# --------------------------

def find_runs(root: Path) -> List[Tuple[str, Path]]:
    """Return list of (run_id, csv_path) discovered under root (recursive)."""
    runs: List[Tuple[str, Path]] = []
    for p in root.rglob("*.CSV"):
        m = RUN_CSV_RE.match(p.name)
        if not m:
            continue
        run_id = m.group(1)
        runs.append((run_id, p))
    runs.sort(key=lambda x: x[0])
    return runs


def read_csv_numeric(csv_path: Path) -> Tuple[List[str], List[Dict[str, float]]]:
    """
    Read CSV skipping metadata lines starting with '#'.
    Returns (columns, rows as dict-of-floats where possible; non-numeric -> NaN).
    """
    rows: List[Dict[str, float]] = []
    header: Optional[List[str]] = None

    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        # Skip leading metadata lines starting with '#'
        # But also allow metadata lines interleaved (defensive): ignore any line starting with '#'
        filtered_lines = (line for line in f if not line.lstrip().startswith("#"))

        reader = csv.reader(filtered_lines)
        for rec in reader:
            if not rec or all(not c.strip() for c in rec):
                continue
            header = [h.strip() for h in rec]
            break

        if header is None:
            return [], []

        for rec in reader:
            if not rec or all(not c.strip() for c in rec):
                continue
            # pad/truncate to header length
            if len(rec) < len(header):
                rec = rec + [""] * (len(header) - len(rec))
            elif len(rec) > len(header):
                rec = rec[: len(header)]

            row: Dict[str, float] = {}
            for k, v in zip(header, rec):
                v = v.strip()
                if v == "":
                    row[k] = float("nan")
                else:
                    try:
                        row[k] = float(v)
                    except Exception:
                        row[k] = float("nan")
            rows.append(row)

    return header, rows


def is_monotonic_increasing(vals: List[float]) -> bool:
    last = -math.inf
    for v in vals:
        if math.isnan(v):
            continue
        if v < last:
            return False
        last = v
    return True


def median(vals: List[float]) -> Optional[float]:
    xs = [v for v in vals if not math.isnan(v)]
    if len(xs) < 3:
        return None
    xs.sort()
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])

def mean(vals: List[float]) -> Optional[float]:
    xs = [v for v in vals if not math.isnan(v)]
    if not xs:
        return None
    return sum(xs) / len(xs)

def stddev(vals: List[float]) -> Optional[float]:
    xs = [v for v in vals if not math.isnan(v)]
    n = len(xs)
    if n < 3:
        return None
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)

def percentile(vals: List[float], p: float) -> Optional[float]:
    xs = [v for v in vals if not math.isnan(v)]
    if not xs:
        return None
    xs.sort()
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1

def nan_fraction(vals: List[float]) -> float:
    if not vals:
        return 1.0
    n_nan = sum(1 for v in vals if math.isnan(v))
    return n_nan / len(vals)


def try_load_json(json_path: Path) -> Tuple[str, Dict]:
    """
    Best-effort JSON load.
    Returns (status, meta) where status in: ok, repaired, invalid.
    """
    txt = json_path.read_text(encoding="utf-8", errors="replace").strip()
    if not txt:
        return "invalid", {}

    # 1) direct parse
    try:
        return "ok", json.loads(txt)
    except Exception:
        pass

    # 2) simple repairs:
    repaired = repair_json_text(txt)
    try:
        return "repaired", json.loads(repaired)
    except Exception:
        return "invalid", {}


def repair_json_text(txt: str) -> str:
    """
    Conservative repair:
    - strip non-printables
    - remove trailing commas before } or ]
    - if truncated, try to close braces/brackets
    """
    # strip non-printables except common whitespace
    cleaned = "".join(ch for ch in txt if (ch.isprintable() or ch in "\r\n\t"))

    # remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # if it looks like it starts with { but doesn't end with }, try to close.
    # Similarly for [ ... ]
    opens = cleaned.count("{")
    closes = cleaned.count("}")
    if opens > closes:
        cleaned = cleaned + ("}" * (opens - closes))

    opens = cleaned.count("[")
    closes = cleaned.count("]")
    if opens > closes:
        cleaned = cleaned + ("]" * (opens - closes))

    return cleaned


def bucket_fs(fs: Optional[float]) -> str:
    if fs is None:
        return "unknown"
    # bucket to nearest integer-ish
    return str(int(round(fs)))


# --------------------------
# QA Logic
# --------------------------

def qa_run(run_id: str, csv_path: Path) -> RunQA:
    r = RunQA(run_id=run_id, csv_path=str(csv_path))

    # CSV parse
    cols, rows = read_csv_numeric(csv_path)
    r.columns = cols
    r.n_rows = len(rows)

    if r.n_rows < MIN_ROWS:
        r.flags.append("too_few_rows")
        r.valid = False

    # core column presence
    missing = [c for c in CORE_COLS if c not in cols]
    r.missing_core_cols = missing
    if missing:
        r.flags.append("missing_core_cols:" + ",".join(missing))
        r.valid = False  # timestamp_ms missing is fatal

    # timebase checks
    if "timestamp_ms" in cols:
        t_ms = [row.get("timestamp_ms", float("nan")) for row in rows]
        r.time_monotonic = is_monotonic_increasing(t_ms)
        if r.time_monotonic is False:
            r.flags.append("time_not_monotonic")
            r.valid = False

        # dt median (seconds)
        # compute diffs ignoring NaNs
        diffs_s: List[float] = []
        last = None
        for v in t_ms:
            if math.isnan(v):
                continue
            if last is not None:
                diffs_s.append((v - last) / 1000.0)
            last = v

        dt_med = median(diffs_s)
        r.dt_median_s = dt_med
        # --- Jitter metrics (ms) ---
diffs_ms = [d * 1000.0 for d in diffs_s if not math.isnan(d)]

if diffs_ms:
    r.dt_min_ms = min(diffs_ms)
    r.dt_max_ms = max(diffs_ms)
    r.dt_std_ms = stddev(diffs_ms)
    r.dt_p10_ms = percentile(diffs_ms, 10.0)
    r.dt_p90_ms = percentile(diffs_ms, 90.0)

    # Warn-only jitter flags
    if r.dt_std_ms is not None and r.dt_std_ms > DT_STD_WARN_MS:
        r.flags.append("dt_jitter_high")

    if r.dt_max_ms is not None and r.dt_max_ms > DT_MAX_WARN_MS:
        r.flags.append("dt_outlier_high")

    if r.dt_min_ms is not None and r.dt_min_ms < DT_MIN_WARN_MS:
        r.flags.append("dt_outlier_low")

        if dt_med is not None and dt_med > 0:
            r.fs_hz_est = 1.0 / dt_med

            if not (DT_MED_MIN <= dt_med <= DT_MED_MAX):
                r.flags.append("odd_sampling_rate")
                # not necessarily invalid, but notable:
                # leave r.valid as-is

    # NaN fraction checks on important numeric columns if present
    for c in QA_NUMERIC_COLS:
        if c in cols:
            vals = [row.get(c, float("nan")) for row in rows]
            frac = nan_fraction(vals)
            r.nan_frac[c] = frac
            if c in CORE_COLS and frac > MAX_NAN_FRAC_CORE:
                r.flags.append(f"excess_nan:{c}")
                r.valid = False

    # JSON sidecar
    json_path = csv_path.with_suffix(".JSON")
    if json_path.exists():
        r.json_path = str(json_path)
        status, meta = try_load_json(json_path)
        r.json_status = status
        r.json_meta = meta if isinstance(meta, dict) else {"_meta": meta}
        if status == "invalid":
            r.flags.append("json_invalid")
            # NOT invalidating run; by design

    return r


def build_report(root: Path) -> Report:
    runs = find_runs(root)
    run_qas: List[RunQA] = []
    cols_union: set[str] = set()
    fs_hist: Dict[str, int] = {}

    for run_id, csv_path in runs:
        rq = qa_run(run_id, csv_path)
        run_qas.append(rq)
        cols_union.update(rq.columns)
        b = bucket_fs(rq.fs_hz_est)
        fs_hist[b] = fs_hist.get(b, 0) + 1

    n_valid = sum(1 for r in run_qas if r.valid)
    n_invalid = len(run_qas) - n_valid

    return Report(
        root=str(root),
        n_runs=len(run_qas),
        n_valid=n_valid,
        n_invalid=n_invalid,
        fs_hz_hist=dict(sorted(fs_hist.items(), key=lambda kv: kv[0])),
        columns_union=sorted(cols_union),
        runs=run_qas,
    )


# --------------------------
# Output
# --------------------------

def write_json_report(rep: Report, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qa_report.json"
    payload = {
        "root": rep.root,
        "n_runs": rep.n_runs,
        "n_valid": rep.n_valid,
        "n_invalid": rep.n_invalid,
        "fs_hz_hist": rep.fs_hz_hist,
        "columns_union": rep.columns_union,
        "runs": [asdict(r) for r in rep.runs],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_csv_report(rep: Report, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qa_report.csv"

fieldnames = [
    "run_id",
    "valid",
    "n_rows",
    "fs_hz_est",
    "dt_median_s",
    "dt_min_ms",
    "dt_max_ms",
    "dt_std_ms",
    "dt_p10_ms",
    "dt_p90_ms",
    "time_monotonic",
    "json_status",
    "flags",
    "csv_path",
    "json_path",
]
with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rep.runs:
            w.writerow(
                {
                    "run_id": r.run_id,
                    "valid": int(bool(r.valid)),
                    "n_rows": r.n_rows,
                    "fs_hz_est": "" if r.fs_hz_est is None else f"{r.fs_hz_est:.2f}",
                    "dt_median_s": "" if r.dt_median_s is None else f"{r.dt_median_s:.6f}",
                    "time_monotonic": "" if r.time_monotonic is None else int(bool(r.time_monotonic)),
                    "json_status": r.json_status,
                    "flags": "|".join(r.flags),
                    "csv_path": r.csv_path,
                    "json_path": r.json_path or "",
                    "dt_min_ms": "" if r.dt_min_ms is None else f"{r.dt_min_ms:.3f}",
                    "dt_max_ms": "" if r.dt_max_ms is None else f"{r.dt_max_ms:.3f}",
                    "dt_std_ms": "" if r.dt_std_ms is None else f"{r.dt_std_ms:.3f}",
                    "dt_p10_ms": "" if r.dt_p10_ms is None else f"{r.dt_p10_ms:.3f}",
                    "dt_p90_ms": "" if r.dt_p90_ms is None else f"{r.dt_p90_ms:.3f}",

                }
            )
return out_path



def print_summary(rep: Report) -> None:
    print("\n=== SD Dump QA Summary ===")
    print(f"Root:      {rep.root}")
    print(f"Runs:      {rep.n_runs}")
    print(f"Valid:     {rep.n_valid}")
    print(f"Invalid:   {rep.n_invalid}")
    print("FS buckets:", rep.fs_hz_hist)

    worst = [r for r in rep.runs if not r.valid]
    if worst:
        print("\nInvalid runs:")
        for r in worst[:20]:
            print(f"  RUN{r.run_id}: flags={r.flags}")

    json_bad = [r for r in rep.runs if r.json_status == "invalid"]
    if json_bad:
        print(f"\nJSON invalid (non-fatal): {len(json_bad)}/{rep.n_runs}")


# --------------------------
# CLI
# --------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Path to unzipped SD dump root")
    ap.add_argument("--out", type=str, default="", help="Output dir (default: <root>/_qa)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser().resolve() if args.out else (root / "_qa")

    rep = build_report(root)
    print_summary(rep)
    jp = write_json_report(rep, out_dir)
    cp = write_csv_report(rep, out_dir)

    print("\nWrote:")
    print(f"  {jp}")
    print(f"  {cp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
