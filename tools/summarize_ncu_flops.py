#!/usr/bin/env python3
"""
Summarize FLOPs from Nsight Compute CSV output.

Primary path:
- Sum all metrics with names starting with `flop_count_` (preferred).

Fallback path (when `flop_count_*` is unavailable):
- Estimate from SASS instruction counters:
  add/mul = 1 FLOP, fma = 2 FLOPs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_float(text: str) -> float | None:
    s = text.strip().replace(",", "")
    if not s:
        return None
    m = NUMBER_RE.search(s)
    if m is None:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def _pick_metric_indices(header: list[str]) -> tuple[int | None, int | None]:
    norm = [h.strip().lower() for h in header]
    name_idx = None
    value_idx = None

    # Prefer explicit NCU columns first.
    for i, h in enumerate(norm):
        if h == "metric name":
            name_idx = i
            break
    for i, h in enumerate(norm):
        if h == "metric value":
            value_idx = i
            break

    # Fallback for variant exports.
    if name_idx is None:
        for i, h in enumerate(norm):
            if "metric" in h and "name" in h:
                name_idx = i
                break
    if value_idx is None:
        for i, h in enumerate(norm):
            if "metric" in h and "value" in h:
                value_idx = i
                break

    return name_idx, value_idx


def read_metric_sums(csv_path: Path) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    metric_name_idx = None
    metric_value_idx = None

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Nsight prefixes runtime log lines; skip them.
            if row[0].startswith("==PROF=="):
                continue

            # Header may appear multiple times in the same CSV.
            if "Metric Name" in row and "Metric Value" in row:
                metric_name_idx = row.index("Metric Name")
                metric_value_idx = row.index("Metric Value")
                continue

            if metric_name_idx is None or metric_value_idx is None:
                continue
            if len(row) <= max(metric_name_idx, metric_value_idx):
                continue

            metric_name = row[metric_name_idx].strip()
            metric_value = _parse_float(row[metric_value_idx])
            if not metric_name or metric_value is None:
                continue
            sums[metric_name] += metric_value

    return dict(sums)


def summarize_flops(metric_sums: dict[str, float]) -> tuple[float | None, dict[str, float], str]:
    flop_count_metrics = {
        name: value
        for name, value in metric_sums.items()
        if name.startswith("flop_count_")
    }
    if flop_count_metrics:
        total = sum(flop_count_metrics.values())
        return total, flop_count_metrics, "from flop_count_* metrics"

    # Fallback: estimate using instruction counters.
    # add/mul = 1 FLOP, fma = 2 FLOPs.
    fallback_patterns = {
        "sp_add": r"_op_fadd_",
        "sp_mul": r"_op_fmul_",
        "sp_fma": r"_op_ffma_",
        "dp_add": r"_op_dadd_",
        "dp_mul": r"_op_dmul_",
        "dp_fma": r"_op_dfma_",
        "hp_add": r"_op_hadd_",
        "hp_mul": r"_op_hmul_",
        "hp_fma": r"_op_hfma_",
    }
    buckets: dict[str, float] = {}
    for key, pat in fallback_patterns.items():
        buckets[key] = sum(v for n, v in metric_sums.items() if re.search(pat, n))

    total = (
        buckets["sp_add"]
        + buckets["sp_mul"]
        + 2.0 * buckets["sp_fma"]
        + buckets["dp_add"]
        + buckets["dp_mul"]
        + 2.0 * buckets["dp_fma"]
        + buckets["hp_add"]
        + buckets["hp_mul"]
        + 2.0 * buckets["hp_fma"]
    )
    if total <= 0:
        return None, buckets, "no usable FLOPs-related metrics found"
    return total, buckets, "estimated from SASS add/mul/fma counters"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FLOPs from ncu CSV.")
    parser.add_argument("--csv", required=True, help="Path to ncu CSV log file.")
    parser.add_argument(
        "--show-top",
        type=int,
        default=20,
        help="Show top-N FLOPs-related metrics by value (default: 20).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    metric_sums = read_metric_sums(csv_path)
    total_flops, used_metrics, note = summarize_flops(metric_sums)

    print("=" * 80)
    print("NCU FLOPs Summary")
    print("=" * 80)
    print(f"csv_path           : {csv_path}")
    print(f"total_metrics      : {len(metric_sums)}")
    print(f"method             : {note}")
    if total_flops is None:
        print("total_flops        : unavailable")
    else:
        print(f"total_flops        : {total_flops:.0f}")
        print(f"total_flops_G      : {total_flops / 1e9:.6f}")
    print("-" * 80)

    items = sorted(used_metrics.items(), key=lambda x: x[1], reverse=True)
    top_n = max(args.show_top, 0)
    if top_n > 0 and items:
        print(f"Top {min(top_n, len(items))} metrics:")
        for name, value in items[:top_n]:
            print(f"{name}: {value}")
    elif not items:
        print("No FLOPs-related metrics extracted.")
    print("=" * 80)


if __name__ == "__main__":
    main()
