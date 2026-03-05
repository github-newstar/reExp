#!/usr/bin/env python3
"""
Run full sliding-window HD95 evaluation for a saved run_name.

Behavior:
- Auto-load config from saved/<run_name>/config.yaml
- Auto-select checkpoint from run dir:
  1) best_model.pth
  2) model_best.pth
  3) latest checkpoint-epoch*.pth
- Force metrics config to src/configs/metrics/brats23_seg_hd95.yaml
- Default partition is test (recommended for final report)
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
from omegaconf import OmegaConf
from scipy.ndimage import label as cc_label
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.data_utils import get_dataloaders
from src.logger import NullWriter
from src.metrics.tracker import MetricTracker
from src.trainer import Trainer
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


def _ensure_test_dataset_cfg(config) -> None:
    if "datasets" not in config:
        raise ValueError("config missing 'datasets' section")

    if "test" not in config.datasets:
        if "val" not in config.datasets:
            raise ValueError("config.datasets has neither 'test' nor 'val'")
        config.datasets.test = OmegaConf.create(
            OmegaConf.to_container(config.datasets.val, resolve=False)
        )

    test_cfg = config.datasets.test
    test_cfg.partition = "test"
    test_cfg.split_strategy = "three_way"
    if "val_ratio" not in test_cfg:
        test_cfg.val_ratio = 0.1
    if "test_ratio" not in test_cfg:
        test_cfg.test_ratio = 0.1

    if "transforms" in config and "instance_transforms" in config.transforms:
        test_cfg.instance_transforms = config.transforms.instance_transforms.inference


_STRUCT26 = np.ones((3, 3, 3), dtype=np.uint8)


def _count_cc_and_fragment_ratio(mask: np.ndarray) -> tuple[int, float]:
    """
    mask: bool ndarray [D,H,W]
    returns:
      - number of connected components (26-connectivity)
      - fragment voxel ratio outside largest component
    """
    labeled, n_cc = cc_label(mask.astype(np.uint8), structure=_STRUCT26)
    if n_cc <= 0:
        return 0, 0.0
    sizes = np.bincount(labeled.ravel())[1:]  # drop background 0
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


def _evaluate_with_cc_fragment_stats(
    trainer: Trainer,
    dataloader,
    part: str,
    pred_threshold: float = 0.5,
):
    trainer.is_train = False
    trainer.current_eval_mode = "full"
    trainer.current_eval_quick_cfg = {}
    trainer.model.eval()

    metric_funcs = trainer.metrics["inference"]
    metrics_tracker = MetricTracker(
        *trainer.config.writer.loss_names,
        *[m.name for m in metric_funcs],
        writer=None,
    )

    cc_counts: dict[str, list[float]] = {"TC": [], "WT": [], "ET": []}
    frag_ratios: dict[str, list[float]] = {"TC": [], "WT": [], "ET": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{part}:full", total=len(dataloader)):
            batch = trainer.process_batch(batch=batch, metrics=metrics_tracker)
            logits = batch["logits"]
            probs = torch.sigmoid(logits)
            pred = (probs > float(pred_threshold)).detach().cpu().numpy().astype(np.uint8)

            num_channels = pred.shape[1]
            for b in range(pred.shape[0]):
                if num_channels > 0:
                    n_cc, frag = _count_cc_and_fragment_ratio(pred[b, 0].astype(bool))
                    cc_counts["TC"].append(float(n_cc))
                    frag_ratios["TC"].append(float(frag))
                if num_channels > 1:
                    n_cc, frag = _count_cc_and_fragment_ratio(pred[b, 1].astype(bool))
                    cc_counts["WT"].append(float(n_cc))
                    frag_ratios["WT"].append(float(frag))
                if num_channels > 2:
                    n_cc, frag = _count_cc_and_fragment_ratio(pred[b, 2].astype(bool))
                    cc_counts["ET"].append(float(n_cc))
                    frag_ratios["ET"].append(float(frag))

    logs = metrics_tracker.result()
    for region_name in ("TC", "WT", "ET"):
        cc_stats = _summary_stats(cc_counts[region_name])
        frag_stats = _summary_stats(frag_ratios[region_name])
        for key, value in cc_stats.items():
            logs[f"CCCount_{region_name}_{key}"] = float(value)
        for key, value in frag_stats.items():
            logs[f"FragRatio_{region_name}_{key}"] = float(value)

    logs["CCFrag_case_count"] = float(
        max(len(cc_counts["TC"]), len(cc_counts["WT"]), len(cc_counts["ET"]))
    )
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HD95 full evaluation for saved/<run_name>."
    )
    parser.add_argument("--run-name", required=True, help="Run directory name under saved/.")
    parser.add_argument(
        "--save-root",
        default="saved",
        help="Root dir containing run folders (default: saved).",
    )
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
    parser.add_argument(
        "--metrics-config",
        default="src/configs/metrics/brats23_seg_hd95.yaml",
        help="Hydra metrics config path with HD95 metrics.",
    )
    parser.add_argument(
        "--usage-ratio",
        type=float,
        default=1.0,
        help="Override datasets.*.usage_ratio for evaluation (default: 1.0).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on (auto/cuda/cpu).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output json path. Default: print-only.",
    )
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary prediction when computing CC/fragment stats (default: 0.5).",
    )
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

    metrics_cfg_path = Path(args.metrics_config)
    if not metrics_cfg_path.is_absolute():
        metrics_cfg_path = ROOT_PATH / metrics_cfg_path
    if not metrics_cfg_path.exists():
        raise FileNotFoundError(f"Metrics config not found: {metrics_cfg_path}")

    config = OmegaConf.load(config_path)
    metrics_cfg = OmegaConf.load(metrics_cfg_path)

    OmegaConf.set_struct(config, False)
    config.metrics = metrics_cfg
    if args.partition == "test":
        _ensure_test_dataset_cfg(config)

    datasets_cfg = config.get("datasets", {})
    for split_name in ("train", "val", "test"):
        split_cfg = datasets_cfg.get(split_name)
        if split_cfg is None:
            continue
        split_cfg.usage_ratio = float(args.usage_ratio)

    if config.trainer.get("ddp") is None:
        config.trainer.ddp = {}
    config.trainer.ddp.enabled = False
    config.trainer.ddp.distributed_eval = False
    config.trainer.auto_resume = False
    config.trainer.resume_from = None
    config.trainer.override = False
    config.trainer.eval_partitions = [str(args.partition)]
    config.trainer.use_sliding_window_inference = True
    config.trainer.validation_policy = {"enabled": False}
    config.trainer.dynamic_eval = {"enabled": False}
    config.writer.mode = "offline"
    OmegaConf.set_struct(config, True)

    patch_monai_numpy_dtype_compat()
    _set_track_meta_false()
    device = _resolve_device(config=config, cli_device=args.device)

    logger = logging.getLogger("hd95_eval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    logger.info("run_name=%s", args.run_name)
    logger.info("partition=%s", args.partition)
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("metrics=%s", metrics_cfg_path)
    logger.info("device=%s", device)

    dataloaders, batch_transforms = get_dataloaders(
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

    model = instantiate(config.model).to(device)
    criterion = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    writer = NullWriter(logger=logger, project_config=OmegaConf.to_container(config))

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=None,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=None,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=False,
        rank=0,
        world_size=1,
        is_distributed=False,
    )

    checkpoint = _load_checkpoint_compat(checkpoint_path=checkpoint_path, device=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    trainer._model_for_state_dict().load_state_dict(state_dict)

    logs = _evaluate_with_cc_fragment_stats(
        trainer=trainer,
        dataloader=dataloaders[str(args.partition)],
        part=str(args.partition),
        pred_threshold=float(args.pred_threshold),
    )

    print("\n=== HD95 Evaluation Results ===")
    print(f"run_name: {args.run_name}")
    print(f"partition: {args.partition}")
    print(f"checkpoint: {checkpoint_path.name}")
    print(f"usage_ratio: {args.usage_ratio}")
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
            "metrics_config": str(metrics_cfg_path),
            "logs": {k: float(v) if isinstance(v, (int, float)) else v for k, v in logs.items()},
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved_json: {out_path}")


if __name__ == "__main__":
    main()
