#!/usr/bin/env python3
"""
Run full sliding-window test evaluation for a saved run_name.

Key behavior:
- Only requires --run-name.
- Auto-select checkpoint from saved/<run_name>/:
  - best_model.pth (preferred)
  - model_best.pth (compatibility fallback)
- Evaluates on full test split (three-way split, test_ratio=0.1, usage_ratio=1.0).
- Uses sliding-window inference.
- Does NOT use TTA.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure "src" package is importable no matter where the script is executed from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _set_track_meta_false():
    try:
        from monai.data.meta_obj import set_track_meta
    except Exception:
        try:
            from monai.data.meta_tensor import set_track_meta
        except Exception:
            return
    set_track_meta(False)


def _find_best_checkpoint(run_dir: Path) -> tuple[Path, str]:
    best_model = run_dir / "best_model.pth"
    if best_model.exists():
        return best_model, "best_model.pth"

    model_best = run_dir / "model_best.pth"
    if model_best.exists():
        return model_best, "model_best.pth"

    raise FileNotFoundError(
        f"No best checkpoint found under '{run_dir}'. "
        "Expected best_model.pth (or model_best.pth for compatibility)."
    )


def _resolve_device(config, cli_device: str) -> str:
    import torch

    if cli_device != "auto":
        return cli_device
    configured = str(config.trainer.get("device", "auto")).lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured


def _load_checkpoint_compat(checkpoint_path: Path, device: str):
    import torch

    try:
        return torch.load(
            str(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(checkpoint_path), map_location=device)


def _ensure_test_dataset_cfg(cfg) -> None:
    from omegaconf import OmegaConf

    if "datasets" not in cfg:
        raise ValueError("config missing 'datasets' section.")

    if "test" not in cfg.datasets:
        if "val" not in cfg.datasets:
            raise ValueError("config.datasets has neither 'test' nor 'val'.")
        cfg.datasets.test = OmegaConf.create(OmegaConf.to_container(cfg.datasets.val, resolve=False))
    test_cfg = cfg.datasets.test
    test_cfg.partition = "test"

    if "transforms" in cfg and "instance_transforms" in cfg.transforms:
        test_cfg.instance_transforms = cfg.transforms.instance_transforms.inference

    # Force full test split semantics required by this script.
    test_cfg.split_strategy = "three_way"
    test_cfg.usage_ratio = 1.0
    if "val_ratio" not in test_cfg:
        test_cfg.val_ratio = 0.1
    if "test_ratio" not in test_cfg:
        test_cfg.test_ratio = 0.1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run full sliding-window test evaluation from saved/<run_name> "
            "with auto best-model checkpoint."
        )
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
        "--device",
        default="auto",
        help="Device to run on (auto/cuda/cpu). Default: auto.",
    )
    args = parser.parse_args()

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import torch

    from src.datasets.data_utils import get_dataloaders
    from src.logger import NullWriter
    from src.trainer import Trainer
    from src.utils.io_utils import ROOT_PATH
    from src.utils.monai_compat import patch_monai_numpy_dtype_compat

    run_dir = ROOT_PATH / args.save_root / args.run_name
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    checkpoint_path, checkpoint_source = _find_best_checkpoint(run_dir)
    config = OmegaConf.load(config_path)

    OmegaConf.set_struct(config, False)
    _ensure_test_dataset_cfg(config)

    # Force single-process full test evaluation with sliding-window inference.
    if config.trainer.get("ddp") is None:
        config.trainer.ddp = {}
    config.trainer.ddp.enabled = False
    config.trainer.ddp.distributed_eval = False
    config.trainer.resume_from = None
    config.trainer.override = False
    config.trainer.eval_partitions = ["test"]
    config.trainer.use_sliding_window_inference = True
    config.trainer.validation_policy = {"enabled": False}
    config.trainer.dynamic_eval = {"enabled": False}
    config.writer.mode = "offline"
    OmegaConf.set_struct(config, True)

    device = _resolve_device(config=config, cli_device=str(args.device).lower())
    patch_monai_numpy_dtype_compat()
    _set_track_meta_false()

    logger = logging.getLogger("full_test")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    logger.info("Loading run config: %s", config_path)
    logger.info("Using checkpoint: %s (%s)", checkpoint_path, checkpoint_source)
    logger.info("Device: %s", device)
    logger.info(
        "Test split setup: split_strategy=three_way, usage_ratio=1.0, val_ratio=%s, test_ratio=%s",
        config.datasets.test.get("val_ratio", "NA"),
        config.datasets.test.get("test_ratio", "NA"),
    )
    logger.info("Inference setup: sliding_window=True, TTA=False")

    dataloaders, batch_transforms = get_dataloaders(
        config=config,
        device=device,
        distributed=False,
        rank=0,
        world_size=1,
        distributed_eval=False,
    )
    if "test" not in dataloaders:
        raise ValueError(
            f"'test' dataloader not found. Available: {list(dataloaders.keys())}"
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
        part="test",
        dataloader=dataloaders["test"],
        eval_mode="full",
        max_batches=None,
    )

    print("\n=== Full SW Test Results (No TTA) ===")
    print(f"run_name: {args.run_name}")
    print(f"checkpoint: {checkpoint_path.name}")
    print(f"checkpoint_source: {checkpoint_source}")
    print("partition: test")
    for key in sorted(logs.keys()):
        print(f"{key}: {logs[key]}")


if __name__ == "__main__":
    main()
