#!/usr/bin/env python3
"""
Profile model metrics from saved/<run_name>.

Metrics:
- parameter count
- inference latency / throughput
- FLOPs (fvcore + torch.profiler when available)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.profile_model_server import (  # noqa: E402
    LogitsOnlyWrapper,
    benchmark_runtime,
    build_model_from_run_name,
    count_params,
    find_best_checkpoint,
    load_checkpoint_if_needed,
    try_fvcore_flops,
    try_torch_profiler_flops,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile params/latency/FLOPs from saved/<run_name>."
    )
    parser.add_argument("--run-name", required=True, help="saved/<run_name>")
    parser.add_argument("--save-root", default="saved")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path; default auto-picks best_model/model_best/latest.",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--input-shape",
        nargs=5,
        type=int,
        default=[1, 4, 96, 96, 96],
        metavar=("N", "C", "D", "H", "W"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--fvcore-custom-ops", action="store_true")
    parser.add_argument("--no-torch-profiler", action="store_true")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output json path. Default: saved/<run_name>/profile_metrics.json",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model, _, run_dir = build_model_from_run_name(
        run_name=args.run_name,
        save_root=args.save_root,
        overrides=args.override,
    )

    if args.checkpoint is None:
        ckpt_path = find_best_checkpoint(run_dir)
    else:
        raw = Path(args.checkpoint).expanduser()
        ckpt_path = raw if raw.is_absolute() else (run_dir / raw)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_checkpoint_if_needed(model, str(ckpt_path))
    wrapper = LogitsOnlyWrapper(model).to(args.device).eval()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    x = torch.randn(*args.input_shape, device=args.device, dtype=dtype)
    if dtype in (torch.float16, torch.bfloat16):
        wrapper = wrapper.to(dtype=dtype)

    total_params, trainable_params = count_params(wrapper)
    total_ms, ms_per_iter, fps, peak_mem_mb = benchmark_runtime(
        wrapper, x, warmup=args.warmup, iters=args.iters, device=args.device
    )
    fv_flops, fv_msg = try_fvcore_flops(
        wrapper, x, use_custom_ops=args.fvcore_custom_ops
    )
    if args.no_torch_profiler:
        pr_flops, pr_msg = None, "disabled by --no-torch-profiler"
    else:
        pr_flops, pr_msg = try_torch_profiler_flops(wrapper, x, args.device)

    result = {
        "run_name": args.run_name,
        "checkpoint": str(ckpt_path),
        "input_shape": list(args.input_shape),
        "device": args.device,
        "dtype": args.dtype,
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "runtime_total_ms": float(total_ms),
        "latency_ms_per_iter": float(ms_per_iter),
        "throughput_iter_s": float(fps),
        "peak_gpu_mem_mb": float(peak_mem_mb),
        "flops_fvcore": None if fv_flops is None else float(fv_flops),
        "flops_fvcore_note": fv_msg,
        "flops_profiler": None if pr_flops is None else float(pr_flops),
        "flops_profiler_note": pr_msg,
    }

    print("=" * 80)
    print("Run Profile Summary")
    print("=" * 80)
    print(f"run_name           : {result['run_name']}")
    print(f"checkpoint         : {result['checkpoint']}")
    print(f"params_total       : {result['params_total']:,}")
    print(f"params_trainable   : {result['params_trainable']:,}")
    print(f"latency_ms_per_iter: {result['latency_ms_per_iter']:.3f}")
    print(f"throughput_iter_s  : {result['throughput_iter_s']:.3f}")
    if args.device.startswith("cuda"):
        print(f"peak_gpu_mem_mb    : {result['peak_gpu_mem_mb']:.2f}")
    if result["flops_fvcore"] is not None:
        print(f"flops_fvcore       : {result['flops_fvcore']:.0f}")
    else:
        print(f"flops_fvcore       : unavailable ({result['flops_fvcore_note']})")
    if result["flops_profiler"] is not None:
        print(f"flops_profiler     : {result['flops_profiler']:.0f}")
    else:
        print(f"flops_profiler     : unavailable ({result['flops_profiler_note']})")
    print("=" * 80)

    if args.json_out is None:
        out_path = run_dir / "profile_metrics.json"
    else:
        out_path = Path(args.json_out).expanduser()
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved_json         : {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
