#!/usr/bin/env python3
"""
Run connected-component and fragment-ratio statistics from saved/<run_name>.

This script does NOT depend on HD95 metrics config.
It only performs inference and reports:
1) Connected component count statistics (mean/median/max)
2) Fragment voxel ratio statistics (mean/median/max), where:
   fragment_ratio = (foreground_voxels - largest_component_voxels) / foreground_voxels
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from monai.inferers import sliding_window_inference
from omegaconf import OmegaConf
from scipy.ndimage import label as cc_label
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.data_utils import get_dataloaders
from src.utils.io_utils import ROOT_PATH
from src.utils.monai_compat import patch_monai_numpy_dtype_compat


def _set_track_meta_false() -> None:
    try:
        from monai.data.meta_obj import set_track_meta
    except Exception:
        try:
            from monai.data.meta_tensor import set_track_meta
        except Exception:
            return
    set_track_meta(False)


def _checkpoint_epoch_key(path: Path) -> int:
    match = re.search(r"checkpoint-epoch(\d+)\.pth$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _find_best_checkpoint(run_dir: Path) -> Path:
    for name in ("best_model.pth", "model_best.pth"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate

    candidates = sorted(
        run_dir.glob("checkpoint-epoch*.pth"),
        key=_checkpoint_epoch_key,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in '{run_dir}'. "
            "Expected best_model.pth / model_best.pth / checkpoint-epoch*.pth."
        )
    return candidates[-1]


def _resolve_device(config, cli_device: str) -> str:
    if cli_device != "auto":
        return cli_device
    configured = str(config.trainer.get("device", "auto")).lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured


def _load_checkpoint_compat(checkpoint_path: Path, device: str):
    try:
        return torch.load(
            str(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(checkpoint_path), map_location=device)


def _ensure_eval_dataset_cfg(config, partition: str) -> None:
    if "datasets" not in config:
        raise ValueError("config missing 'datasets' section")

    if partition not in config.datasets:
        if partition == "test" and "val" in config.datasets:
            config.datasets.test = OmegaConf.create(
                OmegaConf.to_container(config.datasets.val, resolve=False)
            )
        else:
            raise ValueError(f"config.datasets missing '{partition}' section")

    part_cfg = config.datasets[partition]
    part_cfg.partition = partition
    part_cfg.split_strategy = "three_way"
    if "val_ratio" not in part_cfg:
        part_cfg.val_ratio = 0.1
    if "test_ratio" not in part_cfg:
        part_cfg.test_ratio = 0.1

    if "transforms" in config and "instance_transforms" in config.transforms:
        part_cfg.instance_transforms = config.transforms.instance_transforms.inference


def _predict_logits(model, image: torch.Tensor) -> torch.Tensor:
    outputs = model(image=image)
    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise ValueError("Model output dict must contain 'logits'.")
        return outputs["logits"]
    if torch.is_tensor(outputs):
        return outputs
    raise ValueError(f"Unsupported model output type: {type(outputs)!r}")


_STRUCT26 = np.ones((3, 3, 3), dtype=np.uint8)


def _count_cc_and_fragment_ratio(mask: np.ndarray) -> tuple[int, float]:
    labeled, n_cc = cc_label(mask.astype(np.uint8), structure=_STRUCT26)
    if n_cc <= 0:
        return 0, 0.0
    sizes = np.bincount(labeled.ravel())[1:]
    if sizes.size == 0:
        return int(n_cc), 0.0
    total = int(sizes.sum())
    largest = int(sizes.max())
    frag_ratio = float((total - largest) / max(total, 1))
    return int(n_cc), frag_ratio


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CC-count and fragment-ratio stats from saved/<run_name>."
    )
    parser.add_argument("--run-name", required=True, help="Run directory name under saved/.")
    parser.add_argument("--save-root", default="saved", help="Root dir containing run folders.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Relative path resolves under saved/<run_name>/.",
    )
    parser.add_argument(
        "--partition",
        default="test",
        choices=["val", "test"],
        help="Evaluation partition (default: test).",
    )
    parser.add_argument("--device", default="auto", help="Device to run on (auto/cuda/cpu).")
    parser.add_argument(
        "--usage-ratio",
        type=float,
        default=1.0,
        help="Override datasets.*.usage_ratio for evaluation (default: 1.0).",
    )
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for prediction channels (default: 0.5).",
    )
    parser.add_argument(
        "--no-sliding-window",
        action="store_true",
        help="Disable sliding-window inference and run direct forward.",
    )
    parser.add_argument("--output-json", default=None, help="Optional output json path.")
    args = parser.parse_args()

    if not (0.0 < float(args.usage_ratio) <= 1.0):
        raise ValueError(f"--usage-ratio must be in (0, 1], got {args.usage_ratio}")
    if not (0.0 < float(args.pred_threshold) < 1.0):
        raise ValueError(f"--pred-threshold must be in (0, 1), got {args.pred_threshold}")

    run_dir = ROOT_PATH / args.save_root / args.run_name
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if args.checkpoint is None:
        checkpoint_path = _find_best_checkpoint(run_dir)
    else:
        raw_ckpt = Path(args.checkpoint).expanduser()
        checkpoint_path = raw_ckpt if raw_ckpt.is_absolute() else run_dir / raw_ckpt
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    _ensure_eval_dataset_cfg(config=config, partition=str(args.partition))
    for split_name in ("train", "val", "test"):
        split_cfg = config.get("datasets", {}).get(split_name)
        if split_cfg is not None:
            split_cfg.usage_ratio = float(args.usage_ratio)

    config.trainer.ddp = {"enabled": False, "distributed_eval": False}
    config.trainer.auto_resume = False
    config.trainer.resume_from = None
    config.trainer.override = False
    config.writer.mode = "offline"
    OmegaConf.set_struct(config, True)

    patch_monai_numpy_dtype_compat()
    _set_track_meta_false()
    device = _resolve_device(config=config, cli_device=args.device)

    logger = logging.getLogger("cc_frag_eval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    logger.info("run_name=%s", args.run_name)
    logger.info("partition=%s", args.partition)
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("device=%s", device)

    dataloaders, _ = get_dataloaders(
        config=config,
        device=device,
        distributed=False,
        rank=0,
        world_size=1,
        distributed_eval=False,
    )
    if args.partition not in dataloaders:
        raise ValueError(
            f"'{args.partition}' dataloader not found. available={list(dataloaders.keys())}"
        )
    dataloader = dataloaders[str(args.partition)]

    model = instantiate(config.model).to(device)
    checkpoint = _load_checkpoint_compat(checkpoint_path=checkpoint_path, device=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    use_sliding_window = not bool(args.no_sliding_window) and bool(
        config.trainer.get("use_sliding_window_inference", True)
    )
    sw_roi_size = tuple(config.trainer.get("sw_roi_size", [96, 96, 96]))
    sw_batch_size = int(config.trainer.get("sw_batch_size", 1))
    sw_overlap = float(config.trainer.get("sw_overlap", 0.5))

    cc_counts: dict[str, list[float]] = {}
    frag_ratios: dict[str, list[float]] = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{args.partition}:cc_frag", total=len(dataloader)):
            image = batch["image"].to(device)
            if use_sliding_window:
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=tuple(int(x) for x in sw_roi_size),
                    sw_batch_size=sw_batch_size,
                    predictor=lambda x: _predict_logits(model, x),
                    overlap=sw_overlap,
                )
            else:
                logits = _predict_logits(model, image)

            probs = torch.sigmoid(logits)
            pred = (probs > float(args.pred_threshold)).detach().cpu().numpy().astype(np.uint8)
            num_channels = pred.shape[1]

            for c in range(num_channels):
                channel_name = ["TC", "WT", "ET"][c] if c < 3 else f"C{c}"
                cc_counts.setdefault(channel_name, [])
                frag_ratios.setdefault(channel_name, [])
                for b in range(pred.shape[0]):
                    n_cc, frag = _count_cc_and_fragment_ratio(pred[b, c].astype(bool))
                    cc_counts[channel_name].append(float(n_cc))
                    frag_ratios[channel_name].append(float(frag))

    logs: dict[str, float] = {}
    for channel_name in sorted(cc_counts.keys()):
        cc_stats = _summary_stats(cc_counts[channel_name])
        frag_stats = _summary_stats(frag_ratios[channel_name])
        for key, value in cc_stats.items():
            logs[f"CCCount_{channel_name}_{key}"] = float(value)
        for key, value in frag_stats.items():
            logs[f"FragRatio_{channel_name}_{key}"] = float(value)
    logs["CCFrag_case_count"] = float(max((len(v) for v in cc_counts.values()), default=0))

    print("\n=== CC/Fragment Evaluation Results ===")
    print(f"run_name: {args.run_name}")
    print(f"partition: {args.partition}")
    print(f"checkpoint: {checkpoint_path.name}")
    print(f"usage_ratio: {args.usage_ratio}")
    print(f"pred_threshold: {args.pred_threshold}")
    print(f"use_sliding_window: {use_sliding_window}")
    if use_sliding_window:
        print(f"sw_roi_size: {list(sw_roi_size)}")
        print(f"sw_batch_size: {sw_batch_size}")
        print(f"sw_overlap: {sw_overlap}")
    for key in sorted(logs.keys()):
        print(f"{key}: {logs[key]}")

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = ROOT_PATH / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": args.run_name,
            "partition": args.partition,
            "checkpoint": str(checkpoint_path),
            "usage_ratio": float(args.usage_ratio),
            "pred_threshold": float(args.pred_threshold),
            "use_sliding_window": bool(use_sliding_window),
            "logs": {k: float(v) for k, v in logs.items()},
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved_json: {out_path}")


if __name__ == "__main__":
    main()
