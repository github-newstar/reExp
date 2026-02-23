#!/usr/bin/env python3
"""
Run theoretical + Nsight Compute inference FLOPs from run_name.

This script requires only --run-name and prints:
- params_total
- flops_theoretical_G (fvcore with custom op handlers)
- flops_executed_G (from ncu csv summary)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "tools" / "profile_model_server.py"
SUMMARIZE_SCRIPT = ROOT / "tools" / "summarize_ncu_flops.py"

PARAMS_RE = re.compile(r"^\s*params_total\s*:\s*([\d,]+)", re.MULTILINE)
FLOPS_THEORY_RE = re.compile(
    r"^\s*flops_fvcore\s*:\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+\(([0-9]+(?:\.[0-9]+)?)\s+G\)",
    re.MULTILINE,
)
FLOPS_THEORY_UNAVAILABLE_RE = re.compile(
    r"^\s*flops_fvcore\s*:\s*unavailable\s*\((.+)\)\s*$", re.MULTILINE
)
FLOPS_EXEC_RE = re.compile(r"^total_flops_G\s*:\s*([0-9.]+)", re.MULTILINE)

NCU_METRICS = ",".join(
    [
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    ]
)


def _run(cmd: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout + proc.stderr


def _extract_or_raise(pattern: re.Pattern[str], text: str, name: str) -> str:
    m = pattern.search(text)
    if not m:
        raise RuntimeError(f"Failed to parse {name} from command output.")
    return m.group(1)


def _extract_theory_or_raise(text: str) -> str:
    m = FLOPS_THEORY_RE.search(text)
    if m:
        return m.group(2)
    m_unavailable = FLOPS_THEORY_UNAVAILABLE_RE.search(text)
    if m_unavailable:
        raise RuntimeError(
            "flops_fvcore is unavailable from profile output: "
            f"{m_unavailable.group(1)}"
        )
    flops_line = next(
        (line.strip() for line in text.splitlines() if "flops_fvcore" in line),
        "<not found>",
    )
    raise RuntimeError(
        "Failed to parse flops_fvcore_G from command output. "
        f"Raw flops line: {flops_line}"
    )


def _theoretical_from_run(
    run_name: str,
    device: str,
    dtype: str,
    input_shape: list[int],
) -> tuple[int, float]:
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
        "0",
        "--iters",
        "1",
        "--no-torch-profiler",
        "--fvcore-custom-ops",
    ]
    out = _run(cmd)

    params_text = _extract_or_raise(PARAMS_RE, out, "params_total")
    theory_text = _extract_theory_or_raise(out)
    return int(params_text.replace(",", "")), float(theory_text)


def _ncu_infer_and_summarize(
    run_name: str,
    device: str,
    dtype: str,
    input_shape: list[int],
    warmup: int,
    iters: int,
    out_dir: Path,
    ncu_bin: str,
) -> tuple[float, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_path = out_dir / f"{run_name}_infer_only.ncu-rep"
    csv_path = out_dir / f"{run_name}_infer_only.csv"

    ncu_cmd = [
        ncu_bin,
        "--target-processes",
        "all",
        "--metrics",
        NCU_METRICS,
        "--export",
        str(rep_path.with_suffix("")),
        sys.executable,
        str(PROFILE_SCRIPT),
        "--run-name",
        run_name,
        "--device",
        device,
        "--dtype",
        dtype,
        "--input-shape",
        *(str(v) for v in input_shape),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--inference-only",
        "--no-torch-profiler",
    ]
    _run(ncu_cmd)

    import_cmd = [ncu_bin, "--import", str(rep_path), "--page", "raw", "--csv"]
    csv_text = _run(import_cmd)
    csv_path.write_text(csv_text, encoding="utf-8")

    sum_cmd = [sys.executable, str(SUMMARIZE_SCRIPT), "--csv", str(csv_path), "--show-top", "12"]
    sum_out = _run(sum_cmd)
    exec_text = _extract_or_raise(FLOPS_EXEC_RE, sum_out, "total_flops_G")
    return float(exec_text), rep_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command theoretical + NCU inference FLOPs from run_name."
    )
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
    parser.add_argument("--warmup", type=int, default=3, help="Warmup forwards in NCU inference run.")
    parser.add_argument("--iters", type=int, default=10, help="Measured forwards in NCU inference run.")
    parser.add_argument("--out-dir", default="ncu_reports", help="Directory to save ncu rep/csv.")
    parser.add_argument("--ncu-bin", default="ncu", help="Nsight Compute executable path.")
    args = parser.parse_args()

    params_total, flops_theoretical_g = _theoretical_from_run(
        run_name=args.run_name,
        device=args.device,
        dtype=args.dtype,
        input_shape=args.input_shape,
    )
    flops_executed_g, rep_path, csv_path = _ncu_infer_and_summarize(
        run_name=args.run_name,
        device=args.device,
        dtype=args.dtype,
        input_shape=args.input_shape,
        warmup=args.warmup,
        iters=args.iters,
        out_dir=Path(args.out_dir).expanduser(),
        ncu_bin=args.ncu_bin,
    )

    ratio = flops_executed_g / flops_theoretical_g if flops_theoretical_g > 0 else 0.0

    print("=" * 80)
    print("FLOPs (Theoretical + NCU Inference)")
    print("=" * 80)
    print(f"run_name           : {args.run_name}")
    print(f"params_total       : {params_total} ({params_total / 1e6:.4f} M)")
    print(f"flops_theoretical_G: {flops_theoretical_g:.6f}")
    print(f"flops_executed_G   : {flops_executed_g:.6f}")
    print(f"executed/theory    : {ratio:.6f}")
    print("-" * 80)
    print(f"ncu_rep_path       : {rep_path}")
    print(f"ncu_csv_path       : {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
