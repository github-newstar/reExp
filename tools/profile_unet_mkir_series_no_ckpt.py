#!/usr/bin/env python3
"""
Profile UNet/MKIR ablation models without checkpoints.

Models:
- 3D UNet
- 3D UNet+DwInception
- 3D UNet+MKIR
- 3D UNet+MKIR+DRBD-Mamba
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.profile_model_server import (  # noqa: E402
    LogitsOnlyWrapper,
    benchmark_runtime,
    build_model_from_config_name,
    count_params,
    try_fvcore_flops,
)


MODEL_SPECS = [
    ("3D UNet", "liunet"),
    ("3D UNet+DwInception", "liudwnet"),
    ("3D UNet+MKIR", "liunet_mkir"),
    ("3D UNet+MKIR+DRBD-Mamba", "liunet_mkir_drbd_mamba"),
]


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def _profile_one(
    display_name: str,
    model_key: str,
    base_config: str,
    input_shape: list[int],
    device: str,
    dtype_name: str,
    warmup: int,
    iters: int,
) -> dict:
    start = time.time()
    row = {
        "name": display_name,
        "model_key": model_key,
        "status": "ok",
        "params_total": None,
        "params_trainable": None,
        "flops_fvcore": None,
        "flops_fvcore_g": None,
        "latency_ms": None,
        "throughput_iter_s": None,
        "peak_gpu_mem_mb": None,
        "error": "",
        "elapsed_s": None,
    }

    wrapper = None
    x = None
    try:
        model, _cfg = build_model_from_config_name(
            config_name=base_config,
            overrides=[f"model={model_key}"],
        )
        wrapper = LogitsOnlyWrapper(model).to(device).eval()
        dtype = _dtype_from_name(dtype_name)
        x = torch.randn(*input_shape, device=device, dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16):
            wrapper = wrapper.to(dtype=dtype)

        params_total, params_trainable = count_params(wrapper)
        _total_ms, latency_ms, throughput, peak_mem_mb = benchmark_runtime(
            wrapper, x, warmup=warmup, iters=iters, device=device
        )
        flops_fvcore, flops_note = try_fvcore_flops(wrapper, x, use_custom_ops=True)

        row["params_total"] = int(params_total)
        row["params_trainable"] = int(params_trainable)
        row["latency_ms"] = float(latency_ms)
        row["throughput_iter_s"] = float(throughput)
        row["peak_gpu_mem_mb"] = float(peak_mem_mb)
        if flops_fvcore is not None:
            row["flops_fvcore"] = float(flops_fvcore)
            row["flops_fvcore_g"] = float(flops_fvcore / 1e9)
        else:
            row["error"] = f"fvcore unavailable: {flops_note}"
    except Exception as exc:  # pragma: no cover
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["elapsed_s"] = round(time.time() - start, 4)
        try:
            del x
            del wrapper
        except Exception:
            pass
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return row


def _fmt_int(value):
    return "-" if value is None else f"{int(value):,}"


def _fmt_float(value, digits=4):
    return "-" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile UNet/MKIR ablation models without checkpoint."
    )
    parser.add_argument(
        "--base-config",
        default="brats23_swin_unetr_cached",
        help="Hydra base config used for model instantiation.",
    )
    parser.add_argument(
        "--input-shape",
        nargs=5,
        type=int,
        default=[1, 4, 96, 96, 96],
        metavar=("N", "C", "D", "H", "W"),
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--out-csv",
        default="saved/profile_unet_mkir_series_no_ckpt.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--out-json",
        default="saved/profile_unet_mkir_series_no_ckpt.json",
        help="JSON output path.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable, fallback to CPU.")
        args.device = "cpu"

    print("=" * 100)
    print("Profiling UNet/MKIR series (random init, no checkpoint).")
    print(f"base_config: {args.base_config}")
    print(f"input_shape: {args.input_shape}")
    print(f"device/dtype: {args.device}/{args.dtype}")
    print(f"warmup/iters: {args.warmup}/{args.iters}")
    print("=" * 100)

    rows = []
    for idx, (name, key) in enumerate(MODEL_SPECS, start=1):
        print(f"[{idx}/{len(MODEL_SPECS)}] {name} ({key}) ...")
        row = _profile_one(
            display_name=name,
            model_key=key,
            base_config=args.base_config,
            input_shape=args.input_shape,
            device=args.device,
            dtype_name=args.dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"  ok | params={_fmt_int(row['params_total'])} | "
                f"flops={_fmt_float(row['flops_fvcore_g'], 4)}G | "
                f"latency={_fmt_float(row['latency_ms'], 3)}ms"
            )
        else:
            print(f"  error | {row['error']}")

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "model_key",
        "status",
        "params_total",
        "params_trainable",
        "flops_fvcore",
        "flops_fvcore_g",
        "latency_ms",
        "throughput_iter_s",
        "peak_gpu_mem_mb",
        "error",
        "elapsed_s",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(
        "| Model | Params | FLOPs(G) | Latency(ms) | Status |\n"
        "|---|---:|---:|---:|---|"
    )
    for row in rows:
        print(
            f"| {row['name']} | {_fmt_int(row['params_total'])} | "
            f"{_fmt_float(row['flops_fvcore_g'], 4)} | "
            f"{_fmt_float(row['latency_ms'], 3)} | {row['status']} |"
        )
    print("=" * 100)
    print(f"CSV : {out_csv}")
    print(f"JSON: {out_json}")
    print("Note: all metrics are measured from randomly initialized weights.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
