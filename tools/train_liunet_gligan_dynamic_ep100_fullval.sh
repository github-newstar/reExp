#!/usr/bin/env bash
set -euo pipefail

# LiuNet + real GliGAN dynamic augmentation
# 100 epochs, batch_size=2, workers=16, warmup=10,
# lr 1e-4 -> 1e-5, full validation 10/5/2, grad clip max=1.

export CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/cached}"
export GLIGAN_DEVICE="${GLIGAN_DEVICE:-cuda}"

python train.py \
  -cn=liunet_gligan_dynamic_cached_ep100_bs2_warm10_lr1e4_1e5_fullval_10_5_2 \
  "$@"
