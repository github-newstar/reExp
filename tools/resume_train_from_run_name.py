#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Ensure "src" package import works no matter where this script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import ROOT_PATH


def _load_checkpoint_compat(checkpoint_path: Path):
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


def _checkpoint_epoch_key(path: Path) -> int:
    match = re.search(r"checkpoint-epoch(\d+)\.pth$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _find_resume_checkpoint(run_dir: Path) -> tuple[Path, str]:
    for name in ("best.pth", "model_best.pth", "best_model.pth"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate, name

    epoch_ckpts = sorted(
        run_dir.glob("checkpoint-epoch*.pth"),
        key=_checkpoint_epoch_key,
    )
    if len(epoch_ckpts) > 0:
        return epoch_ckpts[-1], "latest_epoch_checkpoint"

    raise FileNotFoundError(
        f"No checkpoint found under '{run_dir}'. Expected one of: "
        "best.pth, model_best.pth, best_model.pth, checkpoint-epoch*.pth"
    )


def _read_checkpoint_epoch(checkpoint_path: Path) -> int | None:
    checkpoint = _load_checkpoint_compat(checkpoint_path)
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        return int(checkpoint["epoch"])
    return None


def _build_train_cmd(
    *,
    run_dir: Path,
    run_name: str,
    checkpoint_name: str,
    target_n_epochs: int,
    lr: float,
    batch_size: int,
    max_grad_norm: float,
    lr_policy: str,
) -> list[str]:
    cmd = [
        "python",
        "train.py",
        "--config-path",
        str(run_dir),
        "--config-name",
        "config",
        f"writer.run_name={run_name}",
        "trainer.override=False",
        "trainer.auto_resume=False",
        f"trainer.resume_from={checkpoint_name}",
        f"trainer.n_epochs={target_n_epochs}",
        f"optimizer.lr={lr}",
        f"dataloader.batch_size={batch_size}",
        f"trainer.max_grad_norm={max_grad_norm}",
    ]
    if lr_policy == "constant":
        cmd.extend(
            [
                "lr_scheduler=null",
                "trainer.warmup.enabled=false",
            ]
        )
    else:
        raise ValueError(f"Unsupported --lr-policy: {lr_policy}")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume training by run_name from saved/<run_name>, auto-detect best checkpoint, "
            "and continue for extra epochs with overridden lr/batch_size/max_grad_norm."
        )
    )
    parser.add_argument("--run-name", required=True, help="Run directory name under saved/.")
    parser.add_argument(
        "--save-root",
        default="saved",
        help="Root directory containing run directories (default: saved).",
    )
    parser.add_argument(
        "--extra-epochs",
        type=int,
        default=50,
        help="Additional epochs to train from checkpoint epoch (default: 50).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Override optimizer.lr (default: 2e-5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Override dataloader.batch_size (default: 3).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Override trainer.max_grad_norm (default: 1.0).",
    )
    parser.add_argument(
        "--lr-policy",
        default="constant",
        choices=["constant"],
        help="LR policy for resumed stage (default: constant).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved checkpoint/command without launching training.",
    )
    args = parser.parse_args()

    if args.extra_epochs <= 0:
        raise ValueError(f"--extra-epochs must be > 0, got {args.extra_epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")
    if args.max_grad_norm <= 0:
        raise ValueError(f"--max-grad-norm must be > 0, got {args.max_grad_norm}")

    run_dir = ROOT_PATH / args.save_root / args.run_name
    config_path = run_dir / "config.yaml"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")

    run_cfg = OmegaConf.load(config_path)
    checkpoint_path, checkpoint_source = _find_resume_checkpoint(run_dir)
    checkpoint_epoch = _read_checkpoint_epoch(checkpoint_path)
    if checkpoint_epoch is None:
        checkpoint_epoch = int(run_cfg.trainer.n_epochs)
        print(
            "Warning: checkpoint has no 'epoch' metadata. "
            f"Fallback to config trainer.n_epochs={checkpoint_epoch}."
        )

    target_n_epochs = checkpoint_epoch + int(args.extra_epochs)
    cmd = _build_train_cmd(
        run_dir=run_dir,
        run_name=args.run_name,
        checkpoint_name=checkpoint_path.name,
        target_n_epochs=target_n_epochs,
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        max_grad_norm=float(args.max_grad_norm),
        lr_policy=str(args.lr_policy),
    )

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {checkpoint_path} ({checkpoint_source})")
    print(f"checkpoint_epoch: {checkpoint_epoch}")
    print(f"extra_epochs: {args.extra_epochs}")
    print(f"target_n_epochs: {target_n_epochs}")
    print(f"lr_policy: {args.lr_policy}")
    print("command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
