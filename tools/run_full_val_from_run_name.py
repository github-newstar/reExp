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
    best_path = run_dir / "model_best.pth"
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
    args = parser.parse_args()

    run_dir = ROOT_PATH / args.save_root / args.run_name
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    checkpoint_path = _find_best_checkpoint(run_dir)
    config = OmegaConf.load(config_path)

    OmegaConf.set_struct(config, False)
    # Enforce single-process validation and full sliding-window eval.
    if config.trainer.get("ddp") is None:
        config.trainer.ddp = {}
    config.trainer.ddp.enabled = False
    config.trainer.ddp.distributed_eval = False
    config.trainer.resume_from = None
    config.trainer.override = False
    config.trainer.eval_partitions = ["val"]
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

    dataloaders, batch_transforms = get_dataloaders(
        config=config,
        device=device,
        distributed=False,
        rank=0,
        world_size=1,
        distributed_eval=False,
    )
    if "val" not in dataloaders:
        raise ValueError(
            f"'val' dataloader not found in config datasets: {list(dataloaders.keys())}"
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
        part="val",
        dataloader=dataloaders["val"],
        eval_mode="full",
        max_batches=None,
    )

    print("\n=== Full SW Validation Results ===")
    print(f"run_name: {args.run_name}")
    print(f"checkpoint: {checkpoint_path.name}")
    for key in sorted(logs.keys()):
        print(f"{key}: {logs[key]}")


if __name__ == "__main__":
    main()
