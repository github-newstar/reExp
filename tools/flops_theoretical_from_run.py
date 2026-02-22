#!/usr/bin/env python3
"""
Compute parameter count + theoretical FLOPs from run_name.

This script is a light wrapper around tools/profile_model_server.py and only
requires --run-name.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "tools" / "profile_model_server.py"

PARAMS_RE = re.compile(r"^params_total\s*:\s*([\d,]+)", re.MULTILINE)
FLOPS_RE = re.compile(r"^flops_fvcore\s*:\s*([0-9.]+)\s+\(([0-9.]+)\s+G\)", re.MULTILINE)


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout + proc.stderr


def _extract_params_total(text: str) -> int:
    m = PARAMS_RE.search(text)
    if not m:
        raise RuntimeError("Failed to parse params_total from profile output.")
    return int(m.group(1).replace(",", ""))


def _extract_flops_theoretical_g(text: str) -> float:
    m = FLOPS_RE.search(text)
    if not m:
        raise RuntimeError("Failed to parse flops_fvcore from profile output.")
    return float(m.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Theoretical FLOPs from run_name.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--input-shape",
        nargs=5,
        type=int,
        default=[1, 4, 96, 96, 96],
        metavar=("N", "C", "D", "H", "W"),
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--run-name",
        args.run_name,
        "--input-shape",
        *(str(v) for v in args.input_shape),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--warmup",
        "0",
        "--iters",
        "1",
        "--no-torch-profiler",
        "--fvcore-custom-ops",
    ]
    output = _run(cmd)
    params_total = _extract_params_total(output)
    flops_theory_g = _extract_flops_theoretical_g(output)

    print("=" * 80)
    print("Theoretical FLOPs")
    print("=" * 80)
    print(f"run_name           : {args.run_name}")
    print(f"params_total       : {params_total} ({params_total / 1e6:.4f} M)")
    print(f"flops_theoretical_G: {flops_theory_g:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

