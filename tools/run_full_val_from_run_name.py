#!/usr/bin/env python3
import argparse
import logging
import re
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Ensure "src" package is importable no matter where the script is executed from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.data_utils import get_dataloaders
from src.logger import NullWriter
from src.trainer import Trainer
from src.utils.io_utils import ROOT_PATH
from src.utils.monai_compat import patch_monai_numpy_dtype_compat


def _set_track_meta_false():
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
    # Prefer explicit best checkpoint names first.
    for name in ("best_model.pth", "model_best.pth"):
        best_path = run_dir / name
        if best_path.exists():
            return best_path

    candidates = sorted(
        run_dir.glob("checkpoint-epoch*.pth"),
        key=_checkpoint_epoch_key,
    )
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No checkpoint found under '{run_dir}'. "
            "Expected model_best.pth or checkpoint-epoch*.pth."
        )
    return candidates[-1]


def _resolve_device(config) -> str:
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


def main():
    parser = argparse.ArgumentParser(
        description="Run full sliding-window validation for a saved run_name."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run directory name under saved/, e.g. lgm_aug_bs6_ep300_lr1e4",
    )
    parser.add_argument(
        "--save-root",
        default="saved",
        help="Root directory containing run folders (default: saved).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional checkpoint path. If relative, it will be resolved under "
            "saved/<run_name>/. If not set, auto-select best checkpoint in run dir."
        ),
    )
    parser.add_argument(
        "--partition",
        default="val",
        choices=["val", "test"],
        help="Which partition to evaluate (default: val).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Optional override for datasets.*.val_ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Optional override for datasets.*.test_ratio.",
    )
    parser.add_argument(
        "--last-20-as-val",
        action="store_true",
        help=(
            "Override split to three_way with val_ratio=0.2 and test_ratio=0.0, "
            "then evaluate on val partition."
        ),
    )
    args = parser.parse_args()

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
    if args.last_20_as_val:
        args.partition = "val"
        args.val_ratio = 0.2
        args.test_ratio = 0.0

    if args.val_ratio is not None and not (0.0 <= float(args.val_ratio) < 1.0):
        raise ValueError(f"--val-ratio must be in [0, 1), got {args.val_ratio}")
    if args.test_ratio is not None and not (0.0 <= float(args.test_ratio) < 1.0):
        raise ValueError(f"--test-ratio must be in [0, 1), got {args.test_ratio}")
    if (
        args.val_ratio is not None
        and args.test_ratio is not None
        and float(args.val_ratio) + float(args.test_ratio) >= 1.0
    ):
        raise ValueError(
            f"--val-ratio + --test-ratio must be < 1, got {args.val_ratio + args.test_ratio}"
        )

    datasets_cfg = config.get("datasets", {})
    for split_name in ("train", "val", "test"):
        split_cfg = datasets_cfg.get(split_name)
        if split_cfg is None:
            continue
        if args.val_ratio is not None:
            split_cfg.split_strategy = "three_way"
            split_cfg.val_ratio = float(args.val_ratio)
        if args.test_ratio is not None:
            split_cfg.split_strategy = "three_way"
            split_cfg.test_ratio = float(args.test_ratio)

    # Enforce single-process validation and full sliding-window eval.
    if config.trainer.get("ddp") is None:
        config.trainer.ddp = {}
    config.trainer.ddp.enabled = False
    config.trainer.ddp.distributed_eval = False
    config.trainer.resume_from = None
    config.trainer.override = False
    config.trainer.eval_partitions = [str(args.partition)]
    config.trainer.use_sliding_window_inference = True
    config.trainer.validation_policy = {"enabled": False}
    config.trainer.dynamic_eval = {"enabled": False}
    config.writer.mode = "offline"
    OmegaConf.set_struct(config, True)

    device = _resolve_device(config)
    patch_monai_numpy_dtype_compat()
    _set_track_meta_false()

    logger = logging.getLogger("full_val")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    logger.info("Loading run config: %s", config_path)
    logger.info("Using checkpoint: %s", checkpoint_path)
    logger.info("Device: %s", device)
    logger.info("Evaluation partition: %s", args.partition)
    if args.val_ratio is not None or args.test_ratio is not None:
        logger.info(
            "Split override: val_ratio=%s test_ratio=%s",
            args.val_ratio,
            args.test_ratio,
        )

    dataloaders, batch_transforms = get_dataloaders(
        config=config,
        device=device,
        distributed=False,
        rank=0,
        world_size=1,
        distributed_eval=False,
    )
    part = str(args.partition)
    if part not in dataloaders:
        raise ValueError(
            f"'{part}' dataloader not found in config datasets: {list(dataloaders.keys())}"
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
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    trainer._model_for_state_dict().load_state_dict(state_dict)

    logs = trainer._evaluation_epoch(
        epoch=0,
        part=part,
        dataloader=dataloaders[part],
        eval_mode="full",
        max_batches=None,
    )

    print("\n=== Full SW Validation Results ===")
    print(f"run_name: {args.run_name}")
    print(f"partition: {part}")
    print(f"checkpoint: {checkpoint_path.name}")
    for key in sorted(logs.keys()):
        print(f"{key}: {logs[key]}")


if __name__ == "__main__":
    main()
