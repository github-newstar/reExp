#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch LiuNet training with real GliGAN on-the-fly augmentation."
    )
    parser.add_argument(
        "--config-name",
        default="liunet_cached_ep100_ultrafast_gligan_dynamic",
        help="Hydra config name under src/configs.",
    )
    parser.add_argument(
        "--run-name",
        default="liunet_gligan_dynamic",
        help="writer.run_name override.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="dataloader.num_workers, keep 0 for stable on-the-fly generation.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional trainer.n_epochs override.")
    parser.add_argument("--epoch-len", type=int, default=None, help="Optional trainer.epoch_len override.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="GLIGAN_DEVICE for generator runtime.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional CACHE_DIR override.",
    )
    parser.add_argument(
        "--writer",
        default="wandb",
        choices=["wandb", "noop"],
        help="Writer backend override.",
    )
    return parser.parse_args()


def _must_exist(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def _pick_ckpt(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Missing checkpoint. Tried:\n"
        f"  - {primary}\n"
        f"  - {fallback}"
    )


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    ckpt_t2f = _pick_ckpt(
        root / "pretrained/gligan/weights/brats2023/flair/generator_400000.pt",
        root
        / "pretrained/gligan/weights_raw/Segmentation_Tasks/GliGAN/Checkpoint/brats2023/flair/weights/generator_400000.pt",
    )
    ckpt_t1c = _pick_ckpt(
        root / "pretrained/gligan/weights/brats2023/t1ce/generator_400000.pt",
        root
        / "pretrained/gligan/weights_raw/Segmentation_Tasks/GliGAN/Checkpoint/brats2023/t1ce/weights/generator_400000.pt",
    )
    ckpt_t1n = _pick_ckpt(
        root / "pretrained/gligan/weights/brats2023/t1/generator_400000.pt",
        root
        / "pretrained/gligan/weights_raw/Segmentation_Tasks/GliGAN/Checkpoint/brats2023/t1/weights/generator_400000.pt",
    )
    ckpt_t2w = _pick_ckpt(
        root / "pretrained/gligan/weights/brats2023/t2/generator_400000.pt",
        root
        / "pretrained/gligan/weights_raw/Segmentation_Tasks/GliGAN/Checkpoint/brats2023/t2/weights/generator_400000.pt",
    )
    label_ckpt = _pick_ckpt(
        root / "pretrained/gligan/weights/brats2023/label/G_iter100000.pth",
        root
        / "pretrained/gligan/weights_raw/Segmentation_Tasks/GliGAN/Checkpoint/brats2023/label/weights/G_iter100000.pth",
    )
    for p in [ckpt_t2f, ckpt_t1c, ckpt_t1n, ckpt_t2w]:
        _must_exist(p)
    _must_exist(label_ckpt)

    env = os.environ.copy()
    env["GLIGAN_DEVICE"] = args.device
    env.setdefault("GLIGAN_CKPT_T2F", str(ckpt_t2f))
    env.setdefault("GLIGAN_CKPT_T1C", str(ckpt_t1c))
    env.setdefault("GLIGAN_CKPT_T1N", str(ckpt_t1n))
    env.setdefault("GLIGAN_CKPT_T2W", str(ckpt_t2w))
    env.setdefault("GLIGAN_LABEL_CKPT", str(label_ckpt))
    if args.cache_dir:
        env["CACHE_DIR"] = args.cache_dir

    cmd = [
        sys.executable,
        "train.py",
        f"-cn={args.config_name}",
        f"dataloader.num_workers={args.num_workers}",
        f"writer={args.writer}",
        f"writer.run_name={args.run_name}",
    ]
    if args.epochs is not None:
        cmd.append(f"trainer.n_epochs={int(args.epochs)}")
    if args.epoch_len is not None:
        cmd.append(f"trainer.epoch_len={int(args.epoch_len)}")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
