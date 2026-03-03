#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch LiuNet training with ET copy-paste on-the-fly augmentation."
    )
    parser.add_argument(
        "--config-name",
        default="liunet_cached_ep100_ultrafast_et_copy_paste",
        help="Hydra config name under src/configs.",
    )
    parser.add_argument(
        "--run-name",
        default="liunet_et_copy_paste",
        help="writer.run_name override.",
    )
    parser.add_argument(
        "--donor-bank",
        default="pretrained/et_copy_paste/et_donor_bank.pt",
        help="Path to ET donor bank .pt file.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional dataloader.num_workers override.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional CACHE_DIR override.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional trainer.n_epochs override.",
    )
    parser.add_argument(
        "--writer",
        default="wandb",
        choices=["wandb", "noop"],
        help="Writer backend override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    donor_bank = Path(args.donor_bank).expanduser()
    if not donor_bank.is_absolute():
        donor_bank = (root / donor_bank).resolve()
    if not donor_bank.exists():
        raise FileNotFoundError(f"ET donor bank not found: {donor_bank}")

    env = os.environ.copy()
    env["ET_DONOR_BANK"] = str(donor_bank)
    if args.cache_dir:
        env["CACHE_DIR"] = str(Path(args.cache_dir).expanduser().resolve())

    cmd = [
        sys.executable,
        "train.py",
        f"-cn={args.config_name}",
        f"writer={args.writer}",
        f"writer.run_name={args.run_name}",
    ]
    if args.num_workers is not None:
        cmd.append(f"dataloader.num_workers={int(args.num_workers)}")
    if args.epochs is not None:
        cmd.append(f"trainer.n_epochs={int(args.epochs)}")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
