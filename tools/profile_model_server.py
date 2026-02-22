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
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

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


def build_model_from_config_name(
    config_name: str, overrides: list[str]
) -> tuple[torch.nn.Module, object]:
    with initialize_config_dir(
        version_base=None, config_dir=str(CONFIG_DIR.resolve())
    ):
        cfg = compose(config_name=config_name, overrides=overrides)
    model = instantiate(cfg.model)
    return model, cfg


def _checkpoint_epoch_key(path: Path) -> int:
    name = path.name
    if not name.startswith("checkpoint-epoch") or not name.endswith(".pth"):
        return -1
    epoch_str = name[len("checkpoint-epoch") : -4]
    return int(epoch_str) if epoch_str.isdigit() else -1


def find_best_checkpoint(run_dir: Path) -> Path:
    for name in ("best_model.pth", "model_best.pth"):
        p = run_dir / name
        if p.exists():
            return p
    candidates = sorted(run_dir.glob("checkpoint-epoch*.pth"), key=_checkpoint_epoch_key)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in '{run_dir}'. "
            "Expected best_model.pth/model_best.pth/checkpoint-epoch*.pth."
        )
    return candidates[-1]


def build_model_from_run_name(
    run_name: str,
    save_root: str,
    overrides: list[str],
) -> tuple[torch.nn.Module, object, Path]:
    run_dir = ROOT / save_root / run_name
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    model = instantiate(cfg.model)
    return model, cfg, run_dir


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


def _safe_prod(values: Iterable[int]) -> int | None:
    out = 1
    for v in values:
        if v is None or int(v) <= 0:
            return None
        out *= int(v)
    return out


def _value_shape(value: Any) -> tuple[int, ...] | None:
    try:
        t = value.type()
        if not hasattr(t, "sizes"):
            return None
        sizes = t.sizes()
        if sizes is None:
            return None
        shape = []
        for s in sizes:
            if s is None:
                return None
            si = int(s)
            if si <= 0:
                return None
            shape.append(si)
        return tuple(shape)
    except Exception:
        return None


def _tensor_numel_from_value(value: Any) -> int | None:
    shape = _value_shape(value)
    if shape is None:
        return None
    return _safe_prod(shape)


def _output_numel(outputs: list[Any], inputs: list[Any]) -> int:
    for value in outputs:
        n = _tensor_numel_from_value(value)
        if n is not None:
            return n
    for value in inputs:
        n = _tensor_numel_from_value(value)
        if n is not None:
            return n
    return 0


def _elementwise_flop_handler(op_name: str, cost_per_elem: float):
    def _handler(inputs: list[Any], outputs: list[Any]) -> Counter[str]:
        n = _output_numel(outputs, inputs)
        return Counter({op_name: float(n) * float(cost_per_elem)})

    return _handler


def _fft_rfftn_flop_jit(inputs: list[Any], outputs: list[Any]) -> Counter[str]:
    in_shape = _value_shape(inputs[0]) if inputs else None
    if in_shape is None or len(in_shape) < 3:
        return Counter({"fft_rfftn": 0.0})
    signal_shape = in_shape[-3:]
    signal_n = _safe_prod(signal_shape)
    batch = _safe_prod(in_shape[:-3]) if len(in_shape) > 3 else 1
    if signal_n is None or batch is None:
        return Counter({"fft_rfftn": 0.0})
    # Real FFT complexity uses the common 2.5 * N * log2(N) approximation.
    flops = 2.5 * float(batch) * float(signal_n) * math.log2(max(signal_n, 2))
    return Counter({"fft_rfftn": flops})


def _fft_irfftn_flop_jit(inputs: list[Any], outputs: list[Any]) -> Counter[str]:
    out_shape = _value_shape(outputs[0]) if outputs else None
    if out_shape is None or len(out_shape) < 3:
        return Counter({"fft_irfftn": 0.0})
    signal_shape = out_shape[-3:]
    signal_n = _safe_prod(signal_shape)
    batch = _safe_prod(out_shape[:-3]) if len(out_shape) > 3 else 1
    if signal_n is None or batch is None:
        return Counter({"fft_irfftn": 0.0})
    # Inverse real FFT uses the same asymptotic approximation.
    flops = 2.5 * float(batch) * float(signal_n) * math.log2(max(signal_n, 2))
    return Counter({"fft_irfftn": flops})


def _adaptive_avg_pool3d_flop_jit(inputs: list[Any], outputs: list[Any]) -> Counter[str]:
    in_shape = _value_shape(inputs[0]) if inputs else None
    out_shape = _value_shape(outputs[0]) if outputs else None
    out_numel = _output_numel(outputs, inputs)
    if in_shape is None or out_shape is None or len(in_shape) < 5 or len(out_shape) < 5:
        return Counter({"adaptive_avg_pool3d": float(out_numel)})
    id_, ih, iw = in_shape[-3:]
    od, oh, ow = out_shape[-3:]
    kd = max(id_ // od, 1)
    kh = max(ih // oh, 1)
    kw = max(iw // ow, 1)
    kernel_vol = kd * kh * kw
    return Counter({"adaptive_avg_pool3d": float(out_numel * kernel_vol)})


def _upsample_trilinear3d_flop_jit(inputs: list[Any], outputs: list[Any]) -> Counter[str]:
    out_numel = _output_numel(outputs, inputs)
    # Trilinear interpolation roughly uses 8 weighted samples and 7 accumulations.
    return Counter({"upsample_trilinear3d": float(out_numel) * 15.0})


def _mamba_inner_fn_flop_jit(inputs: list[Any], outputs: list[Any]) -> Counter[str]:
    if len(inputs) < 8:
        return Counter({"mamba_inner_fn": 0.0})
    xz_shape = _value_shape(inputs[0])
    if xz_shape is None or len(xz_shape) < 3:
        return Counter({"mamba_inner_fn": 0.0})
    batch = int(xz_shape[0])
    seq_len = int(xz_shape[-1])
    d_inner = max(int(xz_shape[1]) // 2, 1)

    conv_w_shape = _value_shape(inputs[1]) if len(inputs) > 1 else None
    x_proj_w_shape = _value_shape(inputs[3]) if len(inputs) > 3 else None
    delta_proj_w_shape = _value_shape(inputs[4]) if len(inputs) > 4 else None
    out_proj_w_shape = _value_shape(inputs[5]) if len(inputs) > 5 else None
    a_shape = _value_shape(inputs[7]) if len(inputs) > 7 else None

    d_conv = int(conv_w_shape[-1]) if conv_w_shape and len(conv_w_shape) >= 1 else 0
    d_state = int(a_shape[1]) if a_shape and len(a_shape) >= 2 else 0
    dt_rank = int(delta_proj_w_shape[1]) if delta_proj_w_shape and len(delta_proj_w_shape) >= 2 else 0
    d_model = int(out_proj_w_shape[0]) if out_proj_w_shape and len(out_proj_w_shape) >= 1 else 0
    x_proj_out = int(x_proj_w_shape[0]) if x_proj_w_shape and len(x_proj_w_shape) >= 1 else 0

    flops = 0.0
    # Depth-wise causal conv in the fused kernel.
    if d_conv > 0:
        flops += float(batch * seq_len * d_inner * d_conv)
    # x_proj: (B*L, d_inner) x (d_inner, x_proj_out)
    if x_proj_out > 0:
        flops += float(batch * seq_len * d_inner * x_proj_out)
    # delta_proj: (B*L, dt_rank) x (dt_rank, d_inner)
    if dt_rank > 0:
        flops += float(batch * seq_len * d_inner * dt_rank)
    # Selective scan core (VMamba/Mamba family common approximation).
    if d_state > 0:
        flops += float(9 * batch * seq_len * d_inner * d_state)
    # D and z gates (element-wise terms).
    flops += float(2 * batch * seq_len * d_inner)
    # out_proj: (B*L, d_inner) x (d_inner, d_model)
    if d_model > 0:
        flops += float(batch * seq_len * d_inner * d_model)
    return Counter({"mamba_inner_fn": flops})


def _register_fvcore_custom_op_handles(analysis: Any) -> Any:
    custom_ops = {
        "prim::PythonOp.MambaInnerFn": _mamba_inner_fn_flop_jit,
        "prim::PythonOp.SelectiveScanFn": _mamba_inner_fn_flop_jit,
        "aten::fft_rfftn": _fft_rfftn_flop_jit,
        "aten::fft_irfftn": _fft_irfftn_flop_jit,
        "aten::adaptive_avg_pool3d": _adaptive_avg_pool3d_flop_jit,
        "aten::upsample_trilinear3d": _upsample_trilinear3d_flop_jit,
        "aten::view_as_complex": _elementwise_flop_handler("view_as_complex", 0.0),
        "aten::view_as_real": _elementwise_flop_handler("view_as_real", 0.0),
        "aten::add": _elementwise_flop_handler("add", 1.0),
        "aten::add_": _elementwise_flop_handler("add_", 1.0),
        "aten::mul": _elementwise_flop_handler("mul", 1.0),
        "aten::mul_": _elementwise_flop_handler("mul_", 1.0),
        "aten::neg": _elementwise_flop_handler("neg", 1.0),
        "aten::exp": _elementwise_flop_handler("exp", 1.0),
        "aten::sigmoid": _elementwise_flop_handler("sigmoid", 4.0),
        "aten::silu": _elementwise_flop_handler("silu", 5.0),
        "aten::silu_": _elementwise_flop_handler("silu_", 5.0),
    }
    return analysis.set_op_handle(**custom_ops)


def _unsupported_ops_summary(unsupported: Any, max_items: int = 8) -> str:
    if not unsupported:
        return "none"
    if hasattr(unsupported, "most_common"):
        items = unsupported.most_common(max_items)
    else:
        try:
            items = list(unsupported.items())[:max_items]
        except Exception:
            return str(unsupported)
    return ", ".join(f"{name}:{count}" for name, count in items)


def try_fvcore_flops(
    model: torch.nn.Module, x: torch.Tensor, use_custom_ops: bool = False
) -> tuple[float | None, str]:
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(model, x)
        if use_custom_ops:
            analysis = _register_fvcore_custom_op_handles(analysis)
        flops = analysis.total()
        unsupported = analysis.unsupported_ops()
        msg = "ok"
        if use_custom_ops:
            msg += f"; custom_ops=on; unsupported={_unsupported_ops_summary(unsupported)}"
        return float(flops), msg
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
    parser.add_argument(
        "--config-name",
        default=None,
        help="Hydra config name, e.g. lgmamba_lightfsde_cached_ep300_policy",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Saved run name under save_root. If set, loads saved/<run_name>/config.yaml.",
    )
    parser.add_argument(
        "--save-root",
        default="saved",
        help="Save root for --run-name mode (default: saved).",
    )
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
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional checkpoint .pth path. In --run-name mode, if omitted it auto-picks "
            "best_model.pth/model_best.pth/latest checkpoint from the run directory."
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--fvcore-custom-ops",
        action="store_true",
        help=(
            "Enable custom fvcore op handlers for Mamba/FFT/element-wise ops. "
            "Useful when default fvcore misses FLOPs for unsupported operators."
        ),
    )
    parser.add_argument(
        "--no-torch-profiler",
        action="store_true",
        help=(
            "Disable torch.profiler FLOPs collection. "
            "Use this when running under Nsight Compute (ncu) to avoid CUPTI conflicts."
        ),
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if (args.config_name is None) == (args.run_name is None):
        raise ValueError("Use exactly one of --config-name or --run-name.")

    if args.run_name is not None:
        model, cfg, run_dir = build_model_from_run_name(
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
        checkpoint_str = str(ckpt_path)
    else:
        model, cfg = build_model_from_config_name(
            config_name=args.config_name, overrides=args.override
        )
        checkpoint_str = args.checkpoint
        if checkpoint_str is not None and not Path(checkpoint_str).expanduser().exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_str}")

    load_checkpoint_if_needed(model, checkpoint_str)
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

    fv_flops, fv_msg = try_fvcore_flops(
        wrapper, x, use_custom_ops=args.fvcore_custom_ops
    )

    ncu_related_env = any(
        key in os.environ
        for key in ("NCU_PROFILER_VERSION", "NSIGHT_COMPUTE_MODE", "NV_COMPUTE_PROFILER")
    )
    disable_torch_profiler = args.no_torch_profiler or ncu_related_env
    if disable_torch_profiler:
        pr_flops, pr_msg = None, "disabled (no-torch-profiler or ncu env detected)"
    else:
        pr_flops, pr_msg = try_torch_profiler_flops(wrapper, x, args.device)

    print("=" * 80)
    print("Profile Summary")
    print("=" * 80)
    print(f"config_name        : {args.config_name}")
    print(f"run_name           : {args.run_name}")
    if args.override:
        print(f"overrides          : {args.override}")
    print(f"input_shape        : {_format_shape(args.input_shape)}")
    print(f"device/dtype       : {args.device} / {args.dtype}")
    print(f"checkpoint         : {checkpoint_str}")
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
        if fv_msg != "ok":
            print(f"flops_fvcore_note  : {fv_msg}")
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
