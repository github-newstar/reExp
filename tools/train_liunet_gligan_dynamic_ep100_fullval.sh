#!/usr/bin/env bash
set -euo pipefail

# LiuNet + real GliGAN dynamic augmentation
# 100 epochs, batch_size=2, workers=16, warmup=10,
# lr 1e-4 -> 1e-5, full validation 10/5/2, grad clip max=1.

export CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/cached}"
export GLIGAN_DEVICE="${GLIGAN_DEVICE:-cpu}"

pick_ckpt() {
  local p1="$1"
  local p2="$2"
  if [[ -f "$p1" ]]; then
    echo "$p1"
  elif [[ -f "$p2" ]]; then
    echo "$p2"
  else
    echo "Missing checkpoint. Tried:" >&2
    echo "  - $p1" >&2
    echo "  - $p2" >&2
    exit 1
  fi
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_T2F="$(pick_ckpt \
  "$ROOT_DIR/pretrained/gligan/weights/brats2023/flair/generator_400000.pt" \
  "$ROOT_DIR/pretrained/gligan/weights_raw/Segmentation_Tasks/GliGAN/Checkpoint/brats2023/flair/weights/generator_400000.pt")"
CKPT_T1C="$(pick_ckpt \
  "$ROOT_DIR/pretrained/gligan/weights/brats2023/t1ce/generator_400000.pt" \
  "$ROOT_DIR/pretrained/gligan/weights_raw/Segmentation_Tasks/GliGAN/Checkpoint/brats2023/t1ce/weights/generator_400000.pt")"
CKPT_T1N="$(pick_ckpt \
  "$ROOT_DIR/pretrained/gligan/weights/brats2023/t1/generator_400000.pt" \
  "$ROOT_DIR/pretrained/gligan/weights_raw/Segmentation_Tasks/GliGAN/Checkpoint/brats2023/t1/weights/generator_400000.pt")"
CKPT_T2W="$(pick_ckpt \
  "$ROOT_DIR/pretrained/gligan/weights/brats2023/t2/generator_400000.pt" \
  "$ROOT_DIR/pretrained/gligan/weights_raw/Segmentation_Tasks/GliGAN/Checkpoint/brats2023/t2/weights/generator_400000.pt")"
LABEL_CKPT="$(pick_ckpt \
  "$ROOT_DIR/pretrained/gligan/weights/brats2023/label/G_iter100000.pth" \
  "$ROOT_DIR/pretrained/gligan/weights_raw/Segmentation_Tasks/GliGAN/Checkpoint/brats2023/label/weights/G_iter100000.pth")"

export GLIGAN_CKPT_T2F="$CKPT_T2F"
export GLIGAN_CKPT_T1C="$CKPT_T1C"
export GLIGAN_CKPT_T1N="$CKPT_T1N"
export GLIGAN_CKPT_T2W="$CKPT_T2W"
export GLIGAN_LABEL_CKPT="$LABEL_CKPT"

python train.py \
  -cn=liunet_gligan_dynamic_cached_ep100_bs2_warm10_lr1e4_1e5_fullval_10_5_2 \
  "$@"
