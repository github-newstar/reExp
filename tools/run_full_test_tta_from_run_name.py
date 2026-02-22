#!/usr/bin/env python3
"""
Run full sliding-window test evaluation with 8-flip TTA for a saved run_name.

Key behavior:
- Only requires --run-name.
- Auto-select checkpoint from saved/<run_name>/:
  - best_model.pth (preferred)
  - model_best.pth (compatibility fallback)
- Evaluates on full test split (three-way split, test_ratio=0.1, usage_ratio=1.0).
- Uses sliding-window inference + 8-view flip TTA.
- Supports two Dice aggregation modes:
  - micro: global accumulation over all voxels/cases
  - macro: per-case Dice then average over cases
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
        cfg.datasets.test = OmegaConf.create(
            OmegaConf.to_container(cfg.datasets.val, resolve=False)
        )

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


def _build_eval_dataloader(cfg, part: str):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from src.datasets.collate import collate_fn
    from src.utils.init_utils import set_worker_seed

    dataset = instantiate(cfg.datasets[part])
    dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)

    train_batch_size = int(dataloader_cfg.get("batch_size", 1))
    eval_batch_size = int(dataloader_cfg.get("eval_batch_size", train_batch_size))
    dataloader_cfg.pop("eval_batch_size", None)
    if int(dataloader_cfg.get("num_workers", 0)) <= 0:
        dataloader_cfg.pop("prefetch_factor", None)
        dataloader_cfg["persistent_workers"] = False

    dataloader = instantiate(
        dataloader_cfg,
        dataset=dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=False,
        worker_init_fn=set_worker_seed,
    )
    return dataloader


def _predict_logits(model, image):
    import torch

    outputs = model(image=image)
    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise ValueError("Model output dict must contain 'logits'.")
        return outputs["logits"]
    if torch.is_tensor(outputs):
        return outputs
    raise ValueError(f"Unsupported model output type: {type(outputs)!r}")


def _tta_flip_combinations():
    # Spatial dims for [B, C, D, H, W] are 2,3,4 -> 2^3 = 8 combos
    return [
        (),
        (2,),
        (3,),
        (4,),
        (2, 3),
        (2, 4),
        (3, 4),
        (2, 3, 4),
    ]


def _sliding_window_with_tta(model, image, roi_size, sw_batch_size, overlap):
    from monai.inferers import sliding_window_inference

    combos = _tta_flip_combinations()
    logits_sum = None
    for dims in combos:
        image_in = image.flip(dims=dims) if dims else image
        logits = sliding_window_inference(
            inputs=image_in,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=lambda x: _predict_logits(model, x),
            overlap=overlap,
        )
        if dims:
            logits = logits.flip(dims=dims)
        logits = logits.float()
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / float(len(combos))


def _compute_case_dice_per_channel(pred_channels, label_channels, smooth=1e-5):
    pred = pred_channels.float()
    target = label_channels.float()
    pred_flat = pred.flatten(start_dim=2)
    target_flat = target.flatten(start_dim=2)

    intersection = (pred_flat * target_flat).sum(dim=2)
    pred_sum = pred_flat.sum(dim=2)
    target_sum = target_flat.sum(dim=2)

    return (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)


def _init_micro_accumulator(num_channels: int, device):
    import torch

    return {
        "intersection": torch.zeros(num_channels, dtype=torch.float64, device=device),
        "pred_sum": torch.zeros(num_channels, dtype=torch.float64, device=device),
        "target_sum": torch.zeros(num_channels, dtype=torch.float64, device=device),
        "num_cases": 0,
    }


def _init_macro_accumulator(num_channels: int, device):
    import torch

    return {
        "dice_sum": torch.zeros(num_channels, dtype=torch.float64, device=device),
        "num_cases": 0,
    }


def _update_micro_accumulator(acc, pred_channels, label_channels):
    pred = pred_channels.float()
    target = label_channels.float()
    pred_flat = pred.flatten(start_dim=2)
    target_flat = target.flatten(start_dim=2)
    acc["intersection"] += (pred_flat * target_flat).sum(dim=2).sum(dim=0).double()
    acc["pred_sum"] += pred_flat.sum(dim=2).sum(dim=0).double()
    acc["target_sum"] += target_flat.sum(dim=2).sum(dim=0).double()
    acc["num_cases"] += int(pred.shape[0])


def _update_macro_accumulator(acc, pred_channels, label_channels):
    case_dice = _compute_case_dice_per_channel(pred_channels, label_channels).double()
    acc["dice_sum"] += case_dice.sum(dim=0)
    acc["num_cases"] += int(case_dice.shape[0])


def _finalize_micro_accumulator(acc, smooth=1e-5):
    dice = (2.0 * acc["intersection"] + smooth) / (
        acc["pred_sum"] + acc["target_sum"] + smooth
    )
    mean_dice = float(dice.mean().item()) if dice.numel() > 0 else 0.0
    return {
        "dice_channels": [float(x) for x in dice.tolist()],
        "MeanDice": mean_dice,
        "num_cases": int(acc["num_cases"]),
    }


def _finalize_macro_accumulator(acc):
    if int(acc["num_cases"]) <= 0:
        dice = acc["dice_sum"]
    else:
        dice = acc["dice_sum"] / float(acc["num_cases"])
    mean_dice = float(dice.mean().item()) if dice.numel() > 0 else 0.0
    return {
        "dice_channels": [float(x) for x in dice.tolist()],
        "MeanDice": mean_dice,
        "num_cases": int(acc["num_cases"]),
    }


def _print_metrics_with_names(metrics):
    channel_names = ["Dice_TC", "Dice_WT", "Dice_ET"]
    dice_channels = metrics["dice_channels"]
    for idx, value in enumerate(dice_channels):
        name = channel_names[idx] if idx < len(channel_names) else f"Dice_C{idx}"
        print(f"{name}: {value}")
    print(f"MeanDice: {metrics['MeanDice']}")
    print(f"num_cases: {metrics['num_cases']}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run full sliding-window test evaluation from saved/<run_name> "
            "with auto best-model checkpoint and 8-flip TTA."
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for prediction channels. Default: 0.5.",
    )
    parser.add_argument(
        "--dice-reduction",
        choices=["micro", "macro", "both"],
        default="both",
        help=(
            "Dice aggregation mode: "
            "micro=global voxel accumulation, "
            "macro=per-case mean, both=print both."
        ),
    )
    args = parser.parse_args()

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from tqdm.auto import tqdm
    import torch

    from src.utils.io_utils import ROOT_PATH
    from src.utils.init_utils import set_random_seed
    from src.utils.monai_compat import patch_monai_numpy_dtype_compat

    run_dir = ROOT_PATH / args.save_root / args.run_name
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    checkpoint_path, checkpoint_source = _find_best_checkpoint(run_dir)
    cfg = OmegaConf.load(config_path)

    OmegaConf.set_struct(cfg, False)
    _ensure_test_dataset_cfg(cfg)
    OmegaConf.set_struct(cfg, True)

    set_random_seed(int(cfg.trainer.get("seed", 42)))
    device = _resolve_device(config=cfg, cli_device=str(args.device).lower())

    patch_monai_numpy_dtype_compat()
    _set_track_meta_false()

    logger = logging.getLogger("full_test_tta")
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
        cfg.datasets.test.get("val_ratio", "NA"),
        cfg.datasets.test.get("test_ratio", "NA"),
    )
    logger.info("Inference setup: sliding_window=True, TTA=8-flip")

    dataloader = _build_eval_dataloader(cfg, part="test")
    model = instantiate(cfg.model).to(device)
    model.eval()

    checkpoint = _load_checkpoint_compat(checkpoint_path=checkpoint_path, device=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    roi_size = tuple(int(x) for x in cfg.trainer.get("sw_roi_size", [96, 96, 96]))
    sw_batch_size = int(cfg.trainer.get("sw_batch_size", 1))
    overlap = float(cfg.trainer.get("sw_overlap", 0.5))

    has_label = False
    micro_acc = None
    macro_acc = None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{args.run_name}:test_tta", total=len(dataloader)):
            image = batch["image"].to(device)
            logits = _sliding_window_with_tta(
                model=model,
                image=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
            )
            probs = torch.sigmoid(logits)
            pred_channels = (probs > float(args.threshold)).float()

            label = batch.get("label")
            if label is not None:
                has_label = True
                label = label.to(device).float()
                if micro_acc is None:
                    num_channels = int(pred_channels.shape[1])
                    micro_acc = _init_micro_accumulator(num_channels=num_channels, device=device)
                    macro_acc = _init_macro_accumulator(num_channels=num_channels, device=device)

                _update_micro_accumulator(
                    acc=micro_acc,
                    pred_channels=pred_channels,
                    label_channels=label,
                )
                _update_macro_accumulator(
                    acc=macro_acc,
                    pred_channels=pred_channels,
                    label_channels=label,
                )

    print("\n=== Full SW Test Results (8-flip TTA) ===")
    print(f"run_name: {args.run_name}")
    print(f"checkpoint: {checkpoint_path.name}")
    print(f"checkpoint_source: {checkpoint_source}")
    print("partition: test")
    print(f"threshold: {float(args.threshold)}")
    print(f"sw_roi_size: {list(roi_size)}")
    print(f"sw_batch_size: {sw_batch_size}")
    print(f"sw_overlap: {overlap}")
    print("tta_views: 8")

    if not has_label:
        print("No label found in test batch, metrics unavailable.")
        return

    if args.dice_reduction in {"micro", "both"}:
        micro_metrics = _finalize_micro_accumulator(micro_acc)
        print("[Dice reduction: micro/global-accum]")
        _print_metrics_with_names(micro_metrics)

    if args.dice_reduction in {"macro", "both"}:
        macro_metrics = _finalize_macro_accumulator(macro_acc)
        print("[Dice reduction: macro/case-mean]")
        _print_metrics_with_names(macro_metrics)


if __name__ == "__main__":
    main()
