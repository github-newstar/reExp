#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train UNet3D with ET Copy-Paste and run one-shot HD95 evaluation "
            "after training finishes."
        )
    )
    parser.add_argument(
        "--config-name",
        default="unet3d_cached_ep100_et_copy_paste_fullval",
        help="Hydra config name under src/configs.",
    )
    parser.add_argument(
        "--run-name",
        default="unet3d_cached_ep100_et_copy_paste_fullval",
        help="writer.run_name override.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional CACHE_DIR override.",
    )
    parser.add_argument(
        "--donor-bank",
        default="pretrained/et_copy_paste/et_donor_bank.pt",
        help="Path to ET donor bank .pt file.",
    )
    parser.add_argument(
        "--hd95-partition",
        default="val",
        choices=["val", "test"],
        help="Partition for post-training HD95 evaluation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for HD95 evaluation (auto/cuda/cpu).",
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

    train_cmd = [
        sys.executable,
        "train.py",
        f"-cn={args.config_name}",
        f"writer={args.writer}",
        f"writer.run_name={args.run_name}",
    ]
    hd95_json = f"saved/{args.run_name}/hd95_{args.hd95_partition}_after_train.json"
    hd95_cmd = [
        sys.executable,
        "tools/run_hd95_from_run_name.py",
        "--run-name",
        args.run_name,
        "--partition",
        args.hd95_partition,
        "--device",
        args.device,
        "--output-json",
        hd95_json,
    ]

    print("Running train:", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=root, env=env, check=True)
    print("Running HD95:", " ".join(hd95_cmd))
    subprocess.run(hd95_cmd, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
