#!/usr/bin/env python3
"""
Batch visualization from run_name best checkpoint.

Generates N no-text 3x3 images:
- rows: coronal / sagittal / axial
- cols: T1 modality / GT / prediction

Default output directory: /tmp/pic
Default filenames: sample01.png, sample02.png, ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import ROOT_PATH
from src.utils.monai_compat import patch_monai_numpy_dtype_compat
from tools.visualize_seg_from_run_name import (
    BRATS23_MODALITY_TO_INDEX,
    _best_slice_index,
    _compose_grid,
    _extract_plane,
    _find_best_checkpoint,
    _label_channels_to_display_map,
    _load_checkpoint_compat,
    _pred_channels_to_display_map,
    _predict_channels,
    _resolve_device,
    _robust_norm_image,
    _set_track_meta_false,
    _ensure_eval_partition_cfg,
    _instantiate_dataset,
)


def _select_top_tumor_indices(dataset, num_samples: int) -> list[int]:
    scored: list[tuple[int, int]] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample.get("label")
        if label is None:
            continue
        label_map = _label_channels_to_display_map(torch.as_tensor(label))
        score = int((label_map > 0).sum())
        scored.append((score, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in scored[:num_samples]]


def _pick_slices(gt_map: torch.Tensor | Path | object, pred_map):
    tumor_mask = gt_map > 0
    if tumor_mask.sum() == 0:
        tumor_mask = pred_map > 0
    if tumor_mask.sum() == 0:
        d, h, w = gt_map.shape
        return h // 2, w // 2, d // 2
    idx_cor = _best_slice_index(tumor_mask, axis=1)  # coronal
    idx_sag = _best_slice_index(tumor_mask, axis=2)  # sagittal
    idx_axi = _best_slice_index(tumor_mask, axis=0)  # axial
    return idx_cor, idx_sag, idx_axi


def main():
    parser = argparse.ArgumentParser(
        description="Batch export no-text segmentation figures from run_name."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-root", default="saved")
    parser.add_argument("--partition", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--modality", default="t1n", choices=["t2f", "t1c", "t1n", "t2w"])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="/tmp/pic",
        help="Directory to save output images (default: /tmp/pic).",
    )
    parser.add_argument("--cell-size", type=int, default=320)
    parser.add_argument("--gap", type=int, default=8)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError(f"--num-samples must be > 0, got {args.num_samples}")

    run_dir = ROOT_PATH / args.save_root / args.run_name
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    ckpt_path = _find_best_checkpoint(run_dir)
    cfg = OmegaConf.load(config_path)

    _ensure_eval_partition_cfg(cfg, args.partition)
    device = _resolve_device(cfg, str(args.device).lower())
    patch_monai_numpy_dtype_compat()
    _set_track_meta_false()

    dataset = _instantiate_dataset(cfg, args.partition)
    if len(dataset) == 0:
        raise ValueError(f"Partition '{args.partition}' is empty.")

    selected = _select_top_tumor_indices(dataset, num_samples=min(args.num_samples, len(dataset)))
    if not selected:
        raise ValueError("No valid labeled samples found for visualization.")

    model = instantiate(cfg.model).to(device)
    checkpoint = _load_checkpoint_compat(ckpt_path, device=device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    roi_size = tuple(int(x) for x in cfg.trainer.get("sw_roi_size", [96, 96, 96]))
    sw_batch_size = int(cfg.trainer.get("sw_batch_size", 1))
    overlap = float(cfg.trainer.get("sw_overlap", 0.5))
    modality_idx = BRATS23_MODALITY_TO_INDEX[args.modality]

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["file", "dataset_index", "case_id", "partition", "run_name", "checkpoint"])

        for n, dataset_idx in enumerate(tqdm(selected, desc="Export"), start=1):
            sample = dataset[dataset_idx]
            case_id = str(sample.get("case_id", f"idx{dataset_idx:04d}"))

            image = torch.as_tensor(sample["image"]).float()  # [4,D,H,W]
            label = torch.as_tensor(sample["label"]).float()
            pred_channels = _predict_channels(
                model=model,
                image_4d=image,
                device=device,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                threshold=float(args.threshold),
            )

            image_vol = _robust_norm_image(image[modality_idx].cpu().numpy())
            gt_map = _label_channels_to_display_map(label)
            pred_map = _pred_channels_to_display_map(pred_channels)
            idx_cor, idx_sag, idx_axi = _pick_slices(gt_map, pred_map)

            planes = [
                ("coronal", 1, idx_cor),
                ("sagittal", 2, idx_sag),
                ("axial", 0, idx_axi),
            ]
            planes_image, planes_gt, planes_pred = [], [], []
            for _name, axis, idx in planes:
                planes_image.append(_extract_plane(image_vol, axis=axis, idx=idx))
                planes_gt.append(_extract_plane(gt_map, axis=axis, idx=idx))
                planes_pred.append(_extract_plane(pred_map, axis=axis, idx=idx))

            out_name = f"sample{n:02d}.png"
            out_path = out_dir / out_name
            _compose_grid(
                planes_image=planes_image,
                planes_gt=planes_gt,
                planes_pred=planes_pred,
                out_path=out_path,
                cell_size=int(args.cell_size),
                gap=int(args.gap),
            )
            writer.writerow(
                [out_name, dataset_idx, case_id, args.partition, args.run_name, ckpt_path.name]
            )

    print("Saved dir:", out_dir)
    print("Images   :", len(selected))
    print("Manifest :", manifest_path)
    print("Run      :", args.run_name)
    print("Ckpt     :", ckpt_path.name)


if __name__ == "__main__":
    main()
