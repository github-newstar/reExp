#!/usr/bin/env python3
"""
Collect Chapter-3 metrics for selected models.

Outputs:
1) Structural efficiency metrics from profile_model_server.py
   - params_total
   - flops_fvcore_g
   - latency_ms_per_iter
   - throughput_iter_s
2) Full-test segmentation metrics from run_full_test_from_run_name.py
   - MeanDice (macro)
   - Dice_TC / Dice_WT / Dice_ET (macro)

Typical usage:
  python tools/collect_ch3_metrics.py \
    --run-map "unet3d=RUN_A,liunet=RUN_B,liunet_mkir=RUN_C,liunet_drbd_mamba=RUN_D,liunet_mkir_drbd_mamba=RUN_E" \
    --device cuda --dtype float16 --eval-device cuda
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "tools" / "profile_model_server.py"
EVAL_SCRIPT = ROOT / "tools" / "run_full_test_from_run_name.py"


MODEL_PRESETS: dict[str, dict[str, str]] = {
    "unet3d": {
        "label": "UNet3D (baseline)",
        "config_name": "unet3d_cached_ep100_ultrafast",
    },
    "liunet": {
        "label": "LiuNet",
        "config_name": "liunet_cached_ep100_ultrafast",
    },
    "liunet_mkir": {
        "label": "LiuNet + MKIR",
        "config_name": "liunet_mkir_cached_ep100_bs2_warm10_lr1e4_1e5_fullval_10_5_2",
    },
    "liunet_drbd_mamba": {
        "label": "LiuNet + DRBD-Mamba",
        "config_name": "liunet_drbd_mamba_cached_ep100_bs2_warm10_lr1e4_1e5_fullval_10_5_2",
    },
    "liunet_mkir_drbd_mamba": {
        "label": "LiuNet + MKIR + DRBD-Mamba",
        "config_name": "liunet_mkir_drbd_mamba_cached_ep300_bs6_warm30_lr2e4_1e5_fullval_20_5_2",
    },
}


PARAM_RE = re.compile(r"^\s*params_total\s*:\s*([\d,]+)", re.MULTILINE)
FLOPS_RE = re.compile(
    r"^\s*flops_fvcore\s*:\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+\(([0-9]+(?:\.[0-9]+)?)\s+G\)",
    re.MULTILINE,
)
LAT_RE = re.compile(r"^\s*latency_ms_per_iter\s*:\s*([0-9.]+)\s*$", re.MULTILINE)
TP_RE = re.compile(r"^\s*throughput_iter_s\s*:\s*([0-9.]+)\s*$", re.MULTILINE)
FLOPS_NA_RE = re.compile(r"^\s*flops_fvcore\s*:\s*unavailable\s*\((.+)\)\s*$", re.MULTILINE)

DICE_TC_RE = re.compile(r"^\s*Dice_TC:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
DICE_WT_RE = re.compile(r"^\s*Dice_WT:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
DICE_ET_RE = re.compile(r"^\s*Dice_ET:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
MEAN_DICE_RE = re.compile(r"^\s*MeanDice:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
NUM_CASES_RE = re.compile(r"^\s*num_cases:\s*([0-9]+)\s*$", re.MULTILINE)


def _parse_run_map(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not text.strip():
        return out
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --run-map item: '{item}', expected key=run_name.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in MODEL_PRESETS:
            raise ValueError(
                f"Unknown model key in --run-map: '{key}'. "
                f"Allowed: {', '.join(MODEL_PRESETS.keys())}"
            )
        if not value:
            raise ValueError(f"Empty run_name for model key '{key}'.")
        out[key] = value
    return out


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, text


def _to_float(m: re.Match[str] | None) -> float | None:
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _to_int_commas(m: re.Match[str] | None) -> int | None:
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except Exception:
        return None


def _profile_one(
    *,
    config_name: str,
    input_shape: list[int],
    device: str,
    dtype: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--config-name",
        config_name,
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
    rc, text = _run(cmd)
    out: dict[str, Any] = {}
    if rc != 0:
        out["profile_status"] = "failed"
        out["profile_error"] = f"return_code={rc}"
        out["profile_tail"] = "\n".join(text.splitlines()[-30:])
        return out

    out["profile_status"] = "ok"
    out["params_total"] = _to_int_commas(PARAM_RE.search(text))
    fm = FLOPS_RE.search(text)
    out["flops_fvcore_g"] = float(fm.group(2)) if fm else None
    out["latency_ms_per_iter"] = _to_float(LAT_RE.search(text))
    out["throughput_iter_s"] = _to_float(TP_RE.search(text))
    na = FLOPS_NA_RE.search(text)
    if na and out["flops_fvcore_g"] is None:
        out["flops_note"] = na.group(1).strip()
    return out


def _eval_one(*, run_name: str, save_root: str, device: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--run-name",
        run_name,
        "--save-root",
        save_root,
        "--device",
        device,
        "--dice-reduction",
        "macro",
    ]
    rc, text = _run(cmd)
    out: dict[str, Any] = {}
    if rc != 0:
        out["eval_status"] = "failed"
        out["eval_error"] = f"return_code={rc}"
        out["eval_tail"] = "\n".join(text.splitlines()[-30:])
        return out

    out["eval_status"] = "ok"
    out["Dice_TC"] = _to_float(DICE_TC_RE.search(text))
    out["Dice_WT"] = _to_float(DICE_WT_RE.search(text))
    out["Dice_ET"] = _to_float(DICE_ET_RE.search(text))
    out["MeanDice"] = _to_float(MEAN_DICE_RE.search(text))
    n_match = NUM_CASES_RE.search(text)
    out["num_cases"] = int(n_match.group(1)) if n_match else None
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_md(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "model_key",
        "label",
        "run_name",
        "params_total",
        "flops_fvcore_g",
        "latency_ms_per_iter",
        "throughput_iter_s",
        "MeanDice",
        "Dice_TC",
        "Dice_WT",
        "Dice_ET",
        "num_cases",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("|" + "|".join(headers) + "|\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("|" + "|".join(_fmt(row.get(h)) for h in headers) + "|\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Chapter-3 model metrics (efficiency + test Dice)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_PRESETS.keys()),
        help=f"Model keys to evaluate. Allowed: {', '.join(MODEL_PRESETS.keys())}",
    )
    parser.add_argument(
        "--run-map",
        default="",
        help=(
            "Mapping from model key to saved run_name. "
            "Format: key=run_name,key=run_name"
        ),
    )
    parser.add_argument(
        "--save-root",
        default="saved",
        help="Run directory root for evaluation script (default: saved).",
    )
    parser.add_argument("--device", default="cuda", help="Profile device.")
    parser.add_argument(
        "--eval-device",
        default="cuda",
        help="Evaluation device for run_full_test_from_run_name.py.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Profile dtype.",
    )
    parser.add_argument(
        "--input-shape",
        nargs=5,
        type=int,
        default=[1, 4, 96, 96, 96],
        metavar=("N", "C", "D", "H", "W"),
        help="Profile input shape.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Profile warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Profile measured iterations.")
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip efficiency profiling.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip test-set evaluation.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "tmp" / "ch3_metrics.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "tmp" / "ch3_metrics.md"),
        help="Output markdown table path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_map = _parse_run_map(args.run_map)

    unknown_models = [m for m in args.models if m not in MODEL_PRESETS]
    if unknown_models:
        raise ValueError(
            f"Unknown --models keys: {unknown_models}. "
            f"Allowed: {list(MODEL_PRESETS.keys())}"
        )

    rows: list[dict[str, Any]] = []
    for model_key in args.models:
        preset = MODEL_PRESETS[model_key]
        row: dict[str, Any] = {
            "model_key": model_key,
            "label": preset["label"],
            "config_name": preset["config_name"],
            "run_name": run_map.get(model_key),
        }

        if not args.skip_profile:
            row.update(
                _profile_one(
                    config_name=preset["config_name"],
                    input_shape=args.input_shape,
                    device=args.device,
                    dtype=args.dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )

        if not args.skip_eval:
            run_name = run_map.get(model_key)
            if run_name:
                row.update(
                    _eval_one(
                        run_name=run_name,
                        save_root=args.save_root,
                        device=args.eval_device,
                    )
                )
            else:
                row["eval_status"] = "skipped(no run_name)"

        rows.append(row)

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _write_md(rows, output_md)

    print("=" * 80)
    print("Chapter-3 Metrics Collection Done")
    print("=" * 80)
    print(f"CSV: {output_csv}")
    print(f"MD : {output_md}")
    print("-" * 80)
    print("Models:")
    for row in rows:
        print(
            f"  - {row['model_key']}: "
            f"params={_fmt(row.get('params_total'))}, "
            f"flops={_fmt(row.get('flops_fvcore_g'))}G, "
            f"lat={_fmt(row.get('latency_ms_per_iter'))}ms, "
            f"MeanDice={_fmt(row.get('MeanDice'))}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
