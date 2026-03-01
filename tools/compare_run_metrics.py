#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "tools" / "profile_model_server.py"

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
KV_RE = re.compile(r"^\d{4}-\d{2}-\d{2} .*? - INFO -\s+([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$")
PARAMS_LOG_RE = re.compile(r"^\s*All parameters:\s*([0-9]+)\s*$")

CKPT_EPOCH_RE = re.compile(r"checkpoint-epoch(\d+)\.pth$")

PROFILE_PARAM_RE = re.compile(r"^\s*params_total\s*:\s*([\d,]+)", re.MULTILINE)
PROFILE_FLOPS_RE = re.compile(
    r"^\s*flops_fvcore\s*:\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+\(([0-9]+(?:\.[0-9]+)?)\s+G\)",
    re.MULTILINE,
)
PROFILE_FLOPS_NA_RE = re.compile(r"^\s*flops_fvcore\s*:\s*unavailable\s*\((.+)\)\s*$", re.MULTILINE)
PROFILE_LAT_RE = re.compile(r"^\s*latency_ms_per_iter\s*:\s*([0-9.]+)\s*$", re.MULTILINE)
PROFILE_TP_RE = re.compile(r"^\s*throughput_iter_s\s*:\s*([0-9.]+)\s*$", re.MULTILINE)
PROFILE_MEM_RE = re.compile(r"^\s*peak_gpu_mem_mb\s*:\s*([0-9.]+)\s*$", re.MULTILINE)


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _parse_ts(line: str) -> dt.datetime | None:
    m = TS_RE.match(line)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
    except Exception:
        return None


def _checkpoint_epoch_key(path: Path) -> int:
    m = CKPT_EPOCH_RE.search(path.name)
    if not m:
        return -1
    return int(m.group(1))


def _find_best_checkpoint(run_dir: Path) -> tuple[Path | None, str]:
    for name in ("best_model.pth", "model_best.pth", "best.pth"):
        p = run_dir / name
        if p.exists():
            return p, name
    ckpts = sorted(run_dir.glob("checkpoint-epoch*.pth"), key=_checkpoint_epoch_key)
    if ckpts:
        return ckpts[-1], "latest checkpoint-epoch*.pth"
    return None, "none"


def _find_latest_epoch_checkpoint(run_dir: Path) -> tuple[Path | None, int | None]:
    ckpts = sorted(run_dir.glob("checkpoint-epoch*.pth"), key=_checkpoint_epoch_key)
    if not ckpts:
        return None, None
    return ckpts[-1], _checkpoint_epoch_key(ckpts[-1])


def _load_checkpoint_meta(ckpt_path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"checkpoint_read_error": f"torch import failed: {exc}"}

    try:
        try:
            payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as exc:
        return {"checkpoint_read_error": str(exc)}

    out: dict[str, Any] = {}
    if isinstance(payload, dict):
        if "epoch" in payload:
            out["best_checkpoint_epoch"] = _to_int(str(payload["epoch"]))
        if "monitor_best" in payload:
            mb = payload["monitor_best"]
            out["best_monitor_value"] = float(mb) if mb is not None else None
    return out


def _load_post_full_eval_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "post_full_eval_summary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"post_full_eval_read_error": "invalid json"}
    best = data.get("best_full_eval", {})
    if not isinstance(best, dict):
        return {}
    return {
        "post_full_eval_metric": best.get("metric"),
        "post_full_eval_score": best.get("score"),
        "post_full_eval_epoch": best.get("epoch"),
    }


def _analyze_info_log(info_log: Path) -> dict[str, Any]:
    if not info_log.exists():
        return {"info_log_exists": False}

    first_ts: dt.datetime | None = None
    last_ts: dt.datetime | None = None
    params_total_log: int | None = None
    last_metrics: dict[str, Any] = {}
    last_epoch: int | None = None

    with info_log.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            ts = _parse_ts(line)
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            pm = PARAMS_LOG_RE.match(line.strip())
            if pm:
                params_total_log = int(pm.group(1))

            km = KV_RE.match(line)
            if not km:
                continue
            key = km.group(1)
            value_raw = km.group(2)
            value_num = _to_float(value_raw)
            last_metrics[key] = value_num if value_num is not None else value_raw
            if key == "epoch":
                value_int = _to_int(value_raw)
                if value_int is not None:
                    last_epoch = value_int

    out: dict[str, Any] = {
        "info_log_exists": True,
        "params_total_log": params_total_log,
        "last_logged_epoch": last_epoch,
    }
    if first_ts is not None:
        out["log_start_time"] = first_ts.isoformat(sep=" ")
    if last_ts is not None:
        out["log_end_time"] = last_ts.isoformat(sep=" ")
    if first_ts is not None and last_ts is not None:
        out["train_walltime_hours_from_log"] = (last_ts - first_ts).total_seconds() / 3600.0

    # Keep only metrics relevant for comparisons.
    for k in [
        "TRAIN_MeanDice",
        "VAL_MeanDice",
        "TEST_MeanDice",
        "loss",
        "loss_dice",
        "loss_focal",
        "loss_drbd",
        "grad_norm",
    ]:
        if k in last_metrics:
            out[f"last_{k}"] = last_metrics[k]

    return out


def _run_profile(
    run_name: str,
    *,
    input_shape: list[int],
    device: str,
    dtype: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--run-name",
        run_name,
        "--input-shape",
        *(str(v) for v in input_shape),
        "--device",
        device,
        "--dtype",
        dtype,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--no-torch-profiler",
        "--fvcore-custom-ops",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return {
            "profile_status": "failed",
            "profile_error": f"return_code={proc.returncode}",
            "profile_output_tail": "\n".join(text.splitlines()[-30:]),
        }

    out: dict[str, Any] = {"profile_status": "ok"}
    pm = PROFILE_PARAM_RE.search(text)
    if pm:
        out["params_total_profile"] = int(pm.group(1).replace(",", ""))
    fm = PROFILE_FLOPS_RE.search(text)
    if fm:
        out["flops_fvcore_g"] = float(fm.group(2))
    else:
        fm_na = PROFILE_FLOPS_NA_RE.search(text)
        if fm_na:
            out["flops_fvcore_note"] = fm_na.group(1).strip()

    lm = PROFILE_LAT_RE.search(text)
    if lm:
        out["latency_ms_per_iter"] = float(lm.group(1))
    tm = PROFILE_TP_RE.search(text)
    if tm:
        out["throughput_iter_s"] = float(tm.group(1))
    mm = PROFILE_MEM_RE.search(text)
    if mm:
        out["peak_gpu_mem_mb"] = float(mm.group(1))
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "run_name",
        "params_total",
        "flops_fvcore_g",
        "latency_ms_per_iter",
        "throughput_iter_s",
        "train_walltime_hours_from_log",
        "best_monitor_value",
        "best_checkpoint_epoch",
        "post_full_eval_score",
        "post_full_eval_epoch",
    ]
    line = "|" + "|".join(headers) + "|"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    print(line)
    print(sep)
    for row in rows:
        vals = []
        for h in headers:
            vals.append(_fmt(row.get(h)))
        print("|" + "|".join(vals) + "|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple runs under saved/<run_name> and collect key metrics: "
            "params, inference FLOPs/speed, training walltime (from logs), best score."
        )
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        required=True,
        help="One or more run names under saved/.",
    )
    parser.add_argument(
        "--save-root",
        default="saved",
        help="Directory containing run folders (default: saved).",
    )
    parser.add_argument(
        "--input-shape",
        nargs=5,
        type=int,
        default=[1, 4, 96, 96, 96],
        metavar=("N", "C", "D", "H", "W"),
        help="Inference profiling input shape.",
    )
    parser.add_argument("--device", default="cuda", help="Profiling device.")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Profiling dtype.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Profiling warmup iterations.")
    parser.add_argument("--iters", type=int, default=50, help="Profiling measured iterations.")
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip runtime profiling; only read saved logs/checkpoints.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_root = (ROOT / args.save_root).resolve()
    rows: list[dict[str, Any]] = []

    for run_name in args.run_names:
        row: dict[str, Any] = {"run_name": run_name}
        run_dir = save_root / run_name
        row["run_dir"] = str(run_dir)
        if not run_dir.exists():
            row["status"] = "missing_run_dir"
            rows.append(row)
            continue

        cfg_path = run_dir / "config.yaml"
        row["config_exists"] = cfg_path.exists()

        best_ckpt, best_source = _find_best_checkpoint(run_dir)
        row["best_checkpoint_source"] = best_source
        row["best_checkpoint_path"] = str(best_ckpt) if best_ckpt is not None else None
        if best_ckpt is not None:
            row.update(_load_checkpoint_meta(best_ckpt))

        latest_ckpt, latest_epoch = _find_latest_epoch_checkpoint(run_dir)
        row["latest_checkpoint_path"] = str(latest_ckpt) if latest_ckpt is not None else None
        row["latest_checkpoint_epoch"] = latest_epoch

        row.update(_analyze_info_log(run_dir / "info.log"))
        row.update(_load_post_full_eval_summary(run_dir))

        if not args.skip_profile:
            row.update(
                _run_profile(
                    run_name=run_name,
                    input_shape=list(args.input_shape),
                    device=args.device,
                    dtype=args.dtype,
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                )
            )

        # Prefer profiling params; fallback to log params.
        if row.get("params_total_profile") is not None:
            row["params_total"] = row["params_total_profile"]
        else:
            row["params_total"] = row.get("params_total_log")

        rows.append(row)

    _print_table(rows)

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON: {out_path}")

    if args.output_csv:
        out_path = Path(args.output_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in all_keys:
                    all_keys.append(key)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Saved CSV : {out_path}")


if __name__ == "__main__":
    main()
