#!/usr/bin/env python3
"""
Server-side model profiler for this project.

What it reports:
1) Exact parameter count from model tensors.
2) Runtime latency / throughput on the target device.
3) Peak GPU memory during forward.
4) FLOPs estimates from:
   - fvcore (graph-based, if available)
   - torch.profiler (runtime operator stats, if available)

Notes:
- FLOPs for custom CUDA kernels (e.g., some Mamba ops) may be partially missing
  depending on profiler support. The script prints both estimates when possible.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "src" / "configs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class LogitsOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(image=x)
        if isinstance(out, dict):
            return out["logits"]
        return out


def _format_shape(shape: Iterable[int]) -> str:
    return "x".join(str(int(v)) for v in shape)


def build_model(config_name: str, overrides: list[str]) -> torch.nn.Module:
    with initialize_config_dir(
        version_base=None, config_dir=str(CONFIG_DIR.resolve())
    ):
        cfg = compose(config_name=config_name, overrides=overrides)
    model = instantiate(cfg.model)
    return model


def load_checkpoint_if_needed(model: torch.nn.Module, ckpt_path: str | None) -> None:
    if not ckpt_path:
        return
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_runtime(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    device: str,
) -> tuple[float, float, float, float]:
    model.eval()
    with torch.no_grad():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(warmup):
                _ = model(x)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _ = model(x)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            for _ in range(warmup):
                _ = model(x)
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(x)
            total_ms = (time.perf_counter() - t0) * 1000.0
            peak_mem_mb = 0.0

    ms_per_iter = total_ms / max(iters, 1)
    fps = 1000.0 / ms_per_iter if ms_per_iter > 0 else 0.0
    return total_ms, ms_per_iter, fps, peak_mem_mb


def try_fvcore_flops(model: torch.nn.Module, x: torch.Tensor) -> tuple[float | None, str]:
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, x).total()
        return float(flops), "ok"
    except Exception as e:  # pragma: no cover
        return None, f"{type(e).__name__}: {e}"


def try_torch_profiler_flops(
    model: torch.nn.Module, x: torch.Tensor, device: str
) -> tuple[float | None, str]:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.no_grad():
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_flops=True,
            ) as prof:
                _ = model(x)
        total = 0
        for item in prof.key_averages():
            fl = getattr(item, "flops", 0) or 0
            total += fl
        return float(total), "ok"
    except Exception as e:  # pragma: no cover
        return None, f"{type(e).__name__}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile model params/FLOPs/latency.")
    parser.add_argument("--config-name", required=True, help="Hydra config name, e.g. lgmamba_lightfsde_cached_ep300_policy")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, can be used multiple times. Example: --override model=xxx",
    )
    parser.add_argument(
        "--input-shape",
        nargs=5,
        type=int,
        default=[1, 4, 96, 96, 96],
        metavar=("N", "C", "D", "H", "W"),
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pth")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model = build_model(config_name=args.config_name, overrides=args.override)
    load_checkpoint_if_needed(model, args.checkpoint)
    wrapper = LogitsOnlyWrapper(model).to(args.device).eval()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    x = torch.randn(*args.input_shape, device=args.device, dtype=dtype)

    # Keep model in fp32 by default; only cast model when running low-precision profiling.
    if dtype in (torch.float16, torch.bfloat16):
        wrapper = wrapper.to(dtype=dtype)

    total_params, trainable_params = count_params(wrapper)

    total_ms, ms_per_iter, fps, peak_mem_mb = benchmark_runtime(
        wrapper, x, warmup=args.warmup, iters=args.iters, device=args.device
    )

    fv_flops, fv_msg = try_fvcore_flops(wrapper, x)
    pr_flops, pr_msg = try_torch_profiler_flops(wrapper, x, args.device)

    print("=" * 80)
    print("Profile Summary")
    print("=" * 80)
    print(f"config_name        : {args.config_name}")
    if args.override:
        print(f"overrides          : {args.override}")
    print(f"input_shape        : {_format_shape(args.input_shape)}")
    print(f"device/dtype       : {args.device} / {args.dtype}")
    print(f"checkpoint         : {args.checkpoint}")
    print("-" * 80)
    print(f"params_total       : {total_params:,} ({total_params/1e6:.4f} M)")
    print(f"params_trainable   : {trainable_params:,} ({trainable_params/1e6:.4f} M)")
    print("-" * 80)
    print(f"runtime_total_ms   : {total_ms:.3f} (iters={args.iters}, warmup={args.warmup})")
    print(f"latency_ms_per_iter: {ms_per_iter:.3f}")
    print(f"throughput_iter_s  : {fps:.3f}")
    if args.device.startswith("cuda"):
        print(f"peak_gpu_mem_mb    : {peak_mem_mb:.2f}")
    print("-" * 80)
    if fv_flops is not None:
        print(f"flops_fvcore       : {fv_flops:.0f} ({fv_flops/1e9:.4f} G)")
    else:
        print(f"flops_fvcore       : unavailable ({fv_msg})")
    if pr_flops is not None:
        print(f"flops_profiler     : {pr_flops:.0f} ({pr_flops/1e9:.4f} G)")
    else:
        print(f"flops_profiler     : unavailable ({pr_msg})")
    print("=" * 80)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
