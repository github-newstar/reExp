#!/usr/bin/env python3
"""
Visualize segmentation result from a saved run_name checkpoint.

Output layout (no text on image):
Rows: coronal, sagittal, axial
Cols: T1 modality, ground-truth, prediction
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from monai.inferers import sliding_window_inference
from omegaconf import OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import ROOT_PATH
from src.utils.monai_compat import patch_monai_numpy_dtype_compat


BRATS23_MODALITY_TO_INDEX = {
    "t2f": 0,
    "t1c": 1,
    "t1n": 2,  # native T1
    "t2w": 3,
}


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
    for name in ("best_model.pth", "model_best.pth"):
        p = run_dir / name
        if p.exists():
            return p
    cands = sorted(run_dir.glob("checkpoint-epoch*.pth"), key=_checkpoint_epoch_key)
    if not cands:
        raise FileNotFoundError(
            f"No checkpoint found in {run_dir}. "
            "Expected best_model.pth/model_best.pth/checkpoint-epoch*.pth."
        )
    return cands[-1]


def _load_checkpoint_compat(checkpoint_path: Path, device: str):
    try:
        return torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location=device)


def _resolve_device(cfg, cli_device: str) -> str:
    if cli_device != "auto":
        return cli_device
    configured = str(cfg.trainer.get("device", "auto")).lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured


def _ensure_eval_partition_cfg(cfg, partition: str):
    OmegaConf.set_struct(cfg, False)
    if partition not in cfg.datasets:
        if "val" in cfg.datasets:
            cfg.datasets[partition] = OmegaConf.create(
                OmegaConf.to_container(cfg.datasets["val"], resolve=False)
            )
            cfg.datasets[partition].partition = partition
        else:
            raise ValueError(
                f"Partition '{partition}' not found in config.datasets and no val fallback."
            )

    part_cfg = cfg.datasets[partition]
    part_cfg.partition = partition

    # Always use deterministic inference transforms for visualization.
    if "transforms" in cfg and "instance_transforms" in cfg.transforms:
        part_cfg.instance_transforms = cfg.transforms.instance_transforms.inference

    # Disable expensive val in-memory cache for this one-off visualization call.
    if "cache_in_memory" in part_cfg:
        part_cfg.cache_in_memory = False

    # Ensure test split defaults if absent.
    if part_cfg.get("split_strategy", None) == "three_way":
        if "val_ratio" not in part_cfg:
            part_cfg.val_ratio = 0.1
        if "test_ratio" not in part_cfg:
            part_cfg.test_ratio = 0.1
    OmegaConf.set_struct(cfg, True)


def _instantiate_dataset(cfg, partition: str):
    return instantiate(cfg.datasets[partition])


def _model_logits(model, image: torch.Tensor) -> torch.Tensor:
    out = model(image=image)
    if isinstance(out, dict):
        if "logits" not in out:
            raise ValueError("Model output dict must contain 'logits'.")
        return out["logits"]
    if torch.is_tensor(out):
        return out
    raise ValueError(f"Unsupported model output type: {type(out)!r}")


def _predict_channels(
    model,
    image_4d: torch.Tensor,
    device: str,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    threshold: float,
) -> torch.Tensor:
    image_b = image_4d.unsqueeze(0).to(device)  # [1, C, D, H, W]
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_b,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=lambda x: _model_logits(model, x),
            overlap=overlap,
        )
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()
    return pred.squeeze(0).cpu()  # [3, D, H, W]


def _label_channels_to_display_map(label: torch.Tensor) -> np.ndarray:
    """
    Convert label tensor to display map:
    0=background, 1=WT(gray), 2=ET(white)
    """
    label = torch.as_tensor(label).cpu()
    if label.ndim == 4 and label.shape[0] == 3:
        wt = label[1] > 0.5
        et = label[2] > 0.5
    elif label.ndim == 4 and label.shape[0] == 1:
        scalar = label[0]
        wt = scalar > 0
        et = scalar == 3
    elif label.ndim == 3:
        wt = label > 0
        et = label == 3
    else:
        raise ValueError(f"Unsupported label shape: {tuple(label.shape)}")

    m = torch.zeros_like(wt, dtype=torch.uint8)
    m[wt] = 1
    m[et] = 2
    return m.numpy()


def _pred_channels_to_display_map(pred_channels: torch.Tensor) -> np.ndarray:
    pred = torch.as_tensor(pred_channels).cpu()
    if pred.ndim != 4 or pred.shape[0] < 3:
        raise ValueError(f"Expected pred shape [>=3, D, H, W], got {tuple(pred.shape)}")
    wt = pred[1] > 0.5
    et = pred[2] > 0.5
    m = torch.zeros_like(wt, dtype=torch.uint8)
    m[wt] = 1
    m[et] = 2
    return m.numpy()


def _robust_norm_image(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    nz = vol[np.nonzero(vol)]
    if nz.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo = np.percentile(nz, 1.0)
    hi = np.percentile(nz, 99.0)
    if hi <= lo:
        hi = lo + 1e-6
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return vol


def _largest_tumor_case_index(dataset, scan_limit: int | None = None) -> int:
    n = len(dataset)
    limit = n if (scan_limit is None or scan_limit <= 0) else min(n, scan_limit)
    best_idx = 0
    best_score = -1
    for i in range(limit):
        sample = dataset[i]
        label = sample.get("label")
        if label is None:
            continue
        label_map = _label_channels_to_display_map(torch.as_tensor(label))
        score = int((label_map > 0).sum())
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _best_slice_index(mask3d: np.ndarray, axis: int) -> int:
    axes = [0, 1, 2]
    reduce_axes = tuple(ax for ax in axes if ax != axis)
    area = mask3d.sum(axis=reduce_axes)
    return int(np.argmax(area))


def _extract_plane(vol3d: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:  # axial -> HxW
        sl = vol3d[idx, :, :]
    elif axis == 1:  # coronal -> DxW
        sl = vol3d[:, idx, :]
    elif axis == 2:  # sagittal -> DxH
        sl = vol3d[:, :, idx]
    else:
        raise ValueError(axis)
    # rotate for a consistent visual orientation
    return np.rot90(sl, k=1)


def _to_gray_u8(x2d: np.ndarray) -> np.ndarray:
    x = np.clip(x2d, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _label_to_gray_u8(label2d: np.ndarray) -> np.ndarray:
    out = np.zeros_like(label2d, dtype=np.uint8)
    out[label2d == 1] = 170  # WT
    out[label2d == 2] = 255  # ET
    return out


def _resize_img(arr_u8: np.ndarray, size: int, nearest: bool) -> Image.Image:
    img = Image.fromarray(arr_u8, mode="L")
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return img.resize((size, size), resample=resample)


def _compose_grid(
    planes_image: list[np.ndarray],
    planes_gt: list[np.ndarray],
    planes_pred: list[np.ndarray],
    out_path: Path,
    cell_size: int = 320,
    gap: int = 8,
):
    cols = 3
    rows = 3
    canvas_w = cols * cell_size + (cols + 1) * gap
    canvas_h = rows * cell_size + (rows + 1) * gap
    canvas = Image.new("L", (canvas_w, canvas_h), color=240)  # light gap/background

    for r in range(rows):
        row_imgs = [
            _resize_img(_to_gray_u8(planes_image[r]), cell_size, nearest=False),
            _resize_img(_label_to_gray_u8(planes_gt[r]), cell_size, nearest=True),
            _resize_img(_label_to_gray_u8(planes_pred[r]), cell_size, nearest=True),
        ]
        for c, img in enumerate(row_imgs):
            x0 = gap + c * (cell_size + gap)
            y0 = gap + r * (cell_size + gap)
            canvas.paste(img, (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize segmentation from run_name (no text on output image)."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-root", default="saved")
    parser.add_argument("--partition", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--modality",
        default="t1n",
        choices=["t2f", "t1c", "t1n", "t2w"],
        help="Image modality shown in first column. t1n corresponds to T1.",
    )
    parser.add_argument("--case-id", default=None, help="Optional exact case_id to visualize.")
    parser.add_argument("--sample-index", type=int, default=None, help="Optional sample index.")
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=200,
        help="When case-id/index not given, search top-N samples for largest tumor.",
    )
    parser.add_argument("--cell-size", type=int, default=320)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--out", default=None, help="Output image path (.png).")
    args = parser.parse_args()

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
        raise ValueError(f"Dataset partition '{args.partition}' is empty.")

    if args.case_id is not None:
        selected_index = None
        for i in range(len(dataset)):
            item = dataset[i]
            if str(item.get("case_id", "")) == str(args.case_id):
                selected_index = i
                break
        if selected_index is None:
            raise ValueError(f"case_id '{args.case_id}' not found in partition '{args.partition}'.")
    elif args.sample_index is not None:
        selected_index = int(args.sample_index)
        if selected_index < 0 or selected_index >= len(dataset):
            raise IndexError(
                f"sample-index out of range: {selected_index}, dataset size={len(dataset)}"
            )
    else:
        selected_index = _largest_tumor_case_index(dataset, scan_limit=args.scan_limit)

    sample = dataset[selected_index]
    case_id = str(sample.get("case_id", f"idx{selected_index:04d}"))
    image = torch.as_tensor(sample["image"]).float()  # [4, D, H, W]
    label = sample.get("label")
    if label is None:
        raise ValueError("Selected sample has no label, cannot draw GT column.")
    label_t = torch.as_tensor(label).float()

    model = instantiate(cfg.model).to(device)
    ckpt = _load_checkpoint_compat(ckpt_path, device=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    roi_size = tuple(int(x) for x in cfg.trainer.get("sw_roi_size", [96, 96, 96]))
    sw_batch_size = int(cfg.trainer.get("sw_batch_size", 1))
    overlap = float(cfg.trainer.get("sw_overlap", 0.5))

    pred_channels = _predict_channels(
        model=model,
        image_4d=image,
        device=device,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        threshold=float(args.threshold),
    )

    modality_idx = BRATS23_MODALITY_TO_INDEX[args.modality]
    image_vol = _robust_norm_image(image[modality_idx].cpu().numpy())  # [D,H,W]
    gt_map = _label_channels_to_display_map(label_t)  # [D,H,W] 0/1/2
    pred_map = _pred_channels_to_display_map(pred_channels)  # [D,H,W] 0/1/2

    tumor_mask = gt_map > 0
    if tumor_mask.sum() == 0:
        tumor_mask = pred_map > 0
    if tumor_mask.sum() == 0:
        d, h, w = gt_map.shape
        idx_cor = h // 2
        idx_sag = w // 2
        idx_axi = d // 2
    else:
        idx_cor = _best_slice_index(tumor_mask, axis=1)  # coronal
        idx_sag = _best_slice_index(tumor_mask, axis=2)  # sagittal
        idx_axi = _best_slice_index(tumor_mask, axis=0)  # axial

    planes = [
        ("coronal", 1, idx_cor),
        ("sagittal", 2, idx_sag),
        ("axial", 0, idx_axi),
    ]

    planes_image = []
    planes_gt = []
    planes_pred = []
    for _name, axis, idx in planes:
        planes_image.append(_extract_plane(image_vol, axis=axis, idx=idx))
        planes_gt.append(_extract_plane(gt_map, axis=axis, idx=idx))
        planes_pred.append(_extract_plane(pred_map, axis=axis, idx=idx))

    if args.out is None:
        out_path = run_dir / "visualizations" / f"{case_id}_tri_view.png"
    else:
        out_path = Path(args.out).expanduser().resolve()

    _compose_grid(
        planes_image=planes_image,
        planes_gt=planes_gt,
        planes_pred=planes_pred,
        out_path=out_path,
        cell_size=int(args.cell_size),
        gap=int(args.gap),
    )

    print("Saved:", out_path)
    print("run_name:", args.run_name)
    print("checkpoint:", ckpt_path.name)
    print("partition:", args.partition)
    print("case_id:", case_id)
    print("sample_index:", selected_index)
    print("modality:", args.modality)


if __name__ == "__main__":
    main()
