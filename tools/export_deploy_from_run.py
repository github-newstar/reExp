#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import ROOT_PATH
from src.utils.monai_compat import patch_monai_numpy_dtype_compat


class LogitsOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image=image)
        if isinstance(output, dict):
            if "logits" not in output:
                raise KeyError("Model output dict has no key 'logits'.")
            return output["logits"]
        if not torch.is_tensor(output):
            raise TypeError(f"Unsupported model output type: {type(output)}")
        return output


def _checkpoint_epoch_key(path: Path) -> int:
    match = re.search(r"checkpoint-epoch(\d+)\.pth$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _find_best_checkpoint(run_dir: Path) -> tuple[Path, str]:
    for name in ("best_model.pth", "model_best.pth", "best.pth"):
        path = run_dir / name
        if path.exists():
            return path, name

    candidates = sorted(
        run_dir.glob("checkpoint-epoch*.pth"),
        key=_checkpoint_epoch_key,
    )
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No checkpoint found under '{run_dir}'. "
            "Expected one of: best_model.pth, model_best.pth, best.pth, checkpoint-epoch*.pth."
        )
    return candidates[-1], "latest checkpoint-epoch*.pth"


def _load_checkpoint_compat(checkpoint_path: Path, map_location: str):
    try:
        return torch.load(
            str(checkpoint_path),
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(checkpoint_path), map_location=map_location)


def _extract_state_dict(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    return state_dict


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _resolve_device(device: str) -> str:
    d = str(device).strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to cpu.")
        return "cpu"
    return device


def _parse_input_shape(shape_str: str) -> tuple[int, int, int, int, int]:
    parts = [int(x.strip()) for x in str(shape_str).split(",")]
    if len(parts) != 5:
        raise ValueError(
            f"--input-shape must have 5 ints 'N,C,D,H,W', got: {shape_str!r}"
        )
    if any(v <= 0 for v in parts):
        raise ValueError(f"--input-shape values must be > 0, got: {parts}")
    return tuple(parts)  # type: ignore[return-value]


def _export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
):
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"image": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        export_params=True,
        do_constant_folding=True,
        opset_version=int(opset),
        dynamic_axes=dynamic_axes,
    )


def _export_torchscript(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
):
    traced = torch.jit.trace(model, dummy_input, strict=False)
    traced.save(str(output_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained model from saved/<run_name> to ONNX or TorchScript. "
            "It auto-detects best checkpoint."
        )
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run directory name under saved/, e.g. liunet_xxx.",
    )
    parser.add_argument(
        "--save-root",
        default="saved",
        help="Root containing run folders (default: saved).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional checkpoint path. If relative, resolved under saved/<run_name>/. "
            "If omitted, auto-select best checkpoint."
        ),
    )
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "torchscript", "both"],
        help="Export format (default: onnx).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: saved/<run_name>/deploy.",
    )
    parser.add_argument(
        "--onnx-name",
        default="model.onnx",
        help="Output file name for ONNX (default: model.onnx).",
    )
    parser.add_argument(
        "--torchscript-name",
        default="model.ts",
        help="Output file name for TorchScript (default: model.ts).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device: cpu/cuda/auto (default: cpu).",
    )
    parser.add_argument(
        "--input-shape",
        default="1,4,96,96,96",
        help="Dummy input shape as N,C,D,H,W (default: 1,4,96,96,96).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Enable dynamic batch axis for ONNX export.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading state_dict (default: False).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    patch_monai_numpy_dtype_compat()

    try:
        from monai.data.meta_obj import set_track_meta
    except Exception:
        try:
            from monai.data.meta_tensor import set_track_meta
        except Exception:
            set_track_meta = None
    if set_track_meta is not None:
        set_track_meta(False)

    run_dir = ROOT_PATH / args.save_root / args.run_name
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if args.checkpoint is None:
        checkpoint_path, checkpoint_source = _find_best_checkpoint(run_dir)
    else:
        raw_ckpt = Path(args.checkpoint).expanduser()
        checkpoint_path = raw_ckpt if raw_ckpt.is_absolute() else (run_dir / raw_ckpt)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_source = "custom --checkpoint"

    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model)

    payload = _load_checkpoint_compat(checkpoint_path, map_location="cpu")
    state_dict = _normalize_state_dict_keys(_extract_state_dict(payload))
    load_result = model.load_state_dict(state_dict, strict=bool(args.strict_load))
    if not args.strict_load:
        if len(load_result.missing_keys) > 0:
            print(f"[warn] missing_keys: {len(load_result.missing_keys)}")
        if len(load_result.unexpected_keys) > 0:
            print(f"[warn] unexpected_keys: {len(load_result.unexpected_keys)}")

    export_device = _resolve_device(args.device)
    model = LogitsOnlyWrapper(model).to(export_device).eval()
    input_shape = _parse_input_shape(args.input_shape)
    dummy = torch.randn(*input_shape, device=export_device)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (run_dir / "deploy")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / args.onnx_name
    ts_path = output_dir / args.torchscript_name

    with torch.no_grad():
        if args.format in {"onnx", "both"}:
            _export_onnx(
                model=model,
                dummy_input=dummy,
                output_path=onnx_path,
                opset=int(args.opset),
                dynamic_batch=bool(args.dynamic_batch),
            )

        if args.format in {"torchscript", "both"}:
            _export_torchscript(
                model=model,
                dummy_input=dummy,
                output_path=ts_path,
            )

    print("Export done.")
    print(f"run_name          : {args.run_name}")
    print(f"config            : {config_path}")
    print(f"checkpoint        : {checkpoint_path}")
    print(f"checkpoint_source : {checkpoint_source}")
    print(f"format            : {args.format}")
    print(f"device            : {export_device}")
    print(f"input_shape       : {input_shape}")
    if args.format in {"onnx", "both"}:
        print(f"onnx              : {onnx_path}")
    if args.format in {"torchscript", "both"}:
        print(f"torchscript       : {ts_path}")


if __name__ == "__main__":
    main()
