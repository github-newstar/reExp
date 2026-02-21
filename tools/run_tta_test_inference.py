#!/usr/bin/env python3
"""
Run full sliding-window test inference with 8-flip TTA for top checkpoints
selected from a training run directory.

Usage:
  uv run python tools/run_tta_test_inference.py \
    --run-dir saved/lgm_aug_bs6_ep300_lr1e4 \
    --top-k 5 \
    --reference-data-dir /path/to/BraTS2023
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from hydra.utils import instantiate
from monai.inferers import sliding_window_inference
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# Ensure project root is importable when running as `python tools/...py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_random_seed, set_worker_seed
from src.utils.io_utils import ROOT_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run test sliding-window inference with 8-flip TTA for best checkpoints."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory path or run name under saved/ (e.g. saved/xxx or xxx).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many checkpoints to evaluate (default: 5).",
    )
    parser.add_argument(
        "--ranking-metric",
        type=str,
        default=None,
        help="Metric for ranking checkpoints (default: from post_full_eval_summary.json).",
    )
    parser.add_argument(
        "--part",
        type=str,
        default="test",
        help="Dataset partition to run inference on (default: test).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary channel prediction (default: 0.5).",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable 8-flip TTA (default: enabled).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device, e.g. cuda:0/cpu/auto.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output subfolder under run_dir/tta_test. Default is timestamp.",
    )
    parser.add_argument(
        "--no-write-pt",
        action="store_true",
        help="Do not save segmentation tensor (.pt).",
    )
    parser.add_argument(
        "--no-write-nifti",
        action="store_true",
        help="Do not save segmentation NIfTI (.nii.gz).",
    )
    parser.add_argument(
        "--reference-data-dir",
        type=str,
        default=None,
        help="Raw BraTS root dir for affine/header reference when writing NIfTI.",
    )
    return parser.parse_args()


def resolve_run_dir(run_dir_arg: str) -> Path:
    user_path = Path(run_dir_arg).expanduser()
    if user_path.exists():
        return user_path.resolve()

    saved_path = ROOT_PATH / "saved" / run_dir_arg
    if saved_path.exists():
        return saved_path.resolve()

    raise FileNotFoundError(
        f"run dir not found: {run_dir_arg}. Tried '{user_path}' and '{saved_path}'."
    )


def load_run_config(run_dir: Path):
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in run dir: {config_path}")
    return OmegaConf.load(config_path)


def _infer_epoch_from_checkpoint_name(path: str) -> int | None:
    match = re.search(r"checkpoint-epoch(\d+)\.pth$", str(path))
    if match is None:
        return None
    return int(match.group(1))


def select_checkpoints(run_dir: Path, top_k: int, ranking_metric: str | None):
    summary_path = run_dir / "post_full_eval_summary.json"
    selected = []

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        metric = ranking_metric or summary.get("metric", "val_MeanDice")
        full_results = list(summary.get("full_eval_results", []))
        if full_results:
            ranked = []
            for item in full_results:
                score = item.get("metrics", {}).get(metric)
                if score is None:
                    continue
                checkpoint = item.get("checkpoint")
                if checkpoint is None:
                    continue
                ranked.append((float(score), checkpoint, "full_eval_results"))
            ranked.sort(key=lambda x: x[0], reverse=True)
            for score, checkpoint, source in ranked[:top_k]:
                selected.append(
                    {
                        "checkpoint": checkpoint,
                        "score": score,
                        "source": source,
                        "metric": metric,
                    }
                )
            if selected:
                return selected

        candidates = list(summary.get("candidates", []))
        if candidates:
            ranked = []
            for item in candidates:
                checkpoint = item.get("checkpoint")
                score = item.get("score")
                if checkpoint is None or score is None:
                    continue
                ranked.append((float(score), checkpoint, "candidates"))
            ranked.sort(key=lambda x: x[0], reverse=True)
            for score, checkpoint, source in ranked[:top_k]:
                selected.append(
                    {
                        "checkpoint": checkpoint,
                        "score": score,
                        "source": source,
                        "metric": summary.get("metric", "val_MeanDice"),
                    }
                )
            if selected:
                return selected

    # Fallback: model_best + latest checkpoints
    model_best = run_dir / "model_best.pth"
    if model_best.exists():
        selected.append(
            {
                "checkpoint": str(model_best),
                "score": None,
                "source": "fallback_model_best",
                "metric": None,
            }
        )

    checkpoints = sorted(
        run_dir.glob("checkpoint-epoch*.pth"),
        key=lambda p: _infer_epoch_from_checkpoint_name(str(p)) or -1,
        reverse=True,
    )
    for path in checkpoints:
        if len(selected) >= top_k:
            break
        if str(path) == str(model_best):
            continue
        selected.append(
            {
                "checkpoint": str(path),
                "score": None,
                "source": "fallback_latest",
                "metric": None,
            }
        )
    return selected[:top_k]


def ensure_test_dataset_cfg(cfg, part: str):
    if part in cfg.datasets:
        return
    if "val" not in cfg.datasets:
        raise ValueError(
            f"Partition '{part}' not in run config and cannot derive from val."
        )

    cfg.datasets[part] = copy.deepcopy(cfg.datasets["val"])
    cfg.datasets[part]["partition"] = part
    cfg.datasets[part]["instance_transforms"] = cfg.transforms.instance_transforms.inference


def build_eval_dataloader(cfg, part: str):
    dataset = instantiate(cfg.datasets[part])
    dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)

    train_batch_size = int(dataloader_cfg.get("batch_size", 1))
    eval_batch_size = int(dataloader_cfg.get("eval_batch_size", train_batch_size))
    dataloader_cfg.pop("eval_batch_size", None)
    if int(dataloader_cfg.get("num_workers", 0)) <= 0:
        # torch DataLoader forbids prefetch_factor with num_workers=0
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
    return dataset, dataloader


def _predict_logits(model, image):
    outputs = model(image=image)
    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise ValueError("Model output dict must contain 'logits'.")
        return outputs["logits"]
    if torch.is_tensor(outputs):
        return outputs
    raise ValueError(f"Unsupported model output type: {type(outputs)!r}")


def tta_flip_combinations(enabled: bool):
    if not enabled:
        return [()]
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


def sliding_window_with_tta(
    model,
    image,
    roi_size,
    sw_batch_size,
    overlap,
    tta_enabled,
):
    combos = tta_flip_combinations(enabled=tta_enabled)
    logits_sum = None
    for dims in combos:
        image_in = torch.flip(image, dims=dims) if dims else image
        logits = sliding_window_inference(
            inputs=image_in,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=lambda x: _predict_logits(model, x),
            overlap=overlap,
        )
        if dims:
            logits = torch.flip(logits, dims=dims)
        logits = logits.float()
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / float(len(combos))


def channels_to_brats_label(pred_channels):
    """
    Convert [3, D, H, W] binary channels [TC, WT, ET] to BraTS scalar map.
    """
    if pred_channels.shape[0] != 3:
        raise ValueError(
            f"Expected 3 channels [TC, WT, ET], got {tuple(pred_channels.shape)}"
        )
    tc = pred_channels[0].bool()
    wt = pred_channels[1].bool()
    et = pred_channels[2].bool()

    seg = torch.zeros_like(pred_channels[0], dtype=torch.uint8)
    seg[wt] = 2
    seg[tc] = 1
    seg[et] = 3
    return seg


def build_vector_path_map(dataset):
    mapping = {}
    records = getattr(dataset, "_index", [])
    for record in records:
        case_id = record.get("case_id")
        vector_path = record.get("vector_path")
        if case_id is not None and vector_path is not None:
            mapping[str(case_id)] = str(vector_path)
    return mapping


def load_case_meta(case_id, vector_path_map, meta_cache):
    if case_id in meta_cache:
        return meta_cache[case_id]
    vector_path = vector_path_map.get(case_id)
    if vector_path is None:
        meta_cache[case_id] = None
        return None
    payload = torch.load(vector_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta")
    meta_cache[case_id] = meta
    return meta


def restore_to_orig_shape(seg_cropped, meta):
    if meta is None:
        return seg_cropped
    if not all(k in meta for k in ("orig_shape", "bbox_start", "bbox_end")):
        return seg_cropped

    orig_shape = tuple(int(x) for x in meta["orig_shape"])
    d0, h0, w0 = [int(x) for x in meta["bbox_start"]]
    d1, h1, w1 = [int(x) for x in meta["bbox_end"]]

    full = torch.zeros(orig_shape, dtype=torch.uint8)
    crop_shape = tuple(int(x) for x in seg_cropped.shape)
    expected_shape = (d1 - d0, h1 - h0, w1 - w0)
    if crop_shape != expected_shape:
        # Shape mismatch can happen if additional transforms changed spatial size.
        # In this case keep cropped shape as fallback.
        return seg_cropped
    full[d0:d1, h0:h1, w0:w1] = seg_cropped
    return full


def find_reference_nii(case_id, reference_data_dir: Path | None, cache_dir: Path | None):
    candidates = []
    if reference_data_dir is not None:
        candidates.extend(
            [
                reference_data_dir / case_id / f"{case_id}-t1n.nii.gz",
                reference_data_dir / case_id / f"{case_id}-t1n.nii",
                reference_data_dir / case_id / f"{case_id}-t2f.nii.gz",
                reference_data_dir / case_id / f"{case_id}-t2f.nii",
            ]
        )
    if cache_dir is not None:
        candidates.extend(
            [
                cache_dir / "nifti" / case_id / f"{case_id}-t1n.nii.gz",
                cache_dir / "nifti" / case_id / f"{case_id}-t1n.nii",
                cache_dir / "nifti" / case_id / f"{case_id}-t2f.nii.gz",
                cache_dir / "nifti" / case_id / f"{case_id}-t2f.nii",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def save_nifti(seg, output_path: Path, reference_path: Path):
    import nibabel as nib
    import numpy as np

    ref = nib.load(str(reference_path))
    seg_np = seg.cpu().numpy().astype(np.uint8, copy=False)
    nii = nib.Nifti1Image(seg_np, affine=ref.affine, header=ref.header)
    nib.save(nii, str(output_path))


def init_dice_accumulator(device):
    return {
        "intersection": torch.zeros(3, dtype=torch.float64, device=device),
        "pred_sum": torch.zeros(3, dtype=torch.float64, device=device),
        "target_sum": torch.zeros(3, dtype=torch.float64, device=device),
        "num_cases": 0,
    }


def update_dice_accumulator(acc, pred_channels, label_channels):
    pred = pred_channels.float()
    target = label_channels.float()
    pred_flat = pred.flatten(start_dim=2)
    target_flat = target.flatten(start_dim=2)
    acc["intersection"] += (pred_flat * target_flat).sum(dim=2).sum(dim=0).double()
    acc["pred_sum"] += pred_flat.sum(dim=2).sum(dim=0).double()
    acc["target_sum"] += target_flat.sum(dim=2).sum(dim=0).double()
    acc["num_cases"] += int(pred.shape[0])


def finalize_dice_accumulator(acc, smooth=1e-5):
    numerator = 2.0 * acc["intersection"] + smooth
    denominator = acc["pred_sum"] + acc["target_sum"] + smooth
    dice = numerator / denominator
    return {
        "Dice_TC": float(dice[0].item()),
        "Dice_WT": float(dice[1].item()),
        "Dice_ET": float(dice[2].item()),
        "MeanDice": float(dice.mean().item()),
        "num_cases": int(acc["num_cases"]),
    }


def checkpoint_alias(checkpoint_path: Path):
    name = checkpoint_path.name
    if name == "model_best.pth":
        return "model_best"
    epoch = _infer_epoch_from_checkpoint_name(name)
    if epoch is not None:
        return f"epoch{epoch:04d}"
    return checkpoint_path.stem


def main():
    args = parse_args()
    write_pt = not bool(args.no_write_pt)
    write_nifti = not bool(args.no_write_nifti)
    tta_enabled = not bool(args.no_tta)
    if not write_pt and not write_nifti:
        raise ValueError(
            "Both outputs are disabled. Remove --no-write-pt or --no-write-nifti."
        )
    run_dir = resolve_run_dir(args.run_dir)
    cfg = load_run_config(run_dir)

    set_random_seed(int(cfg.trainer.get("seed", 42)))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ensure_test_dataset_cfg(cfg, args.part)
    dataset, dataloader = build_eval_dataloader(cfg, args.part)
    vector_path_map = build_vector_path_map(dataset)
    meta_cache = {}

    checkpoints = select_checkpoints(
        run_dir=run_dir,
        top_k=max(1, args.top_k),
        ranking_metric=args.ranking_metric,
    )
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found under run dir: {run_dir}")

    output_name = (
        args.output_name
        if args.output_name is not None
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root = run_dir / "tta_test" / output_name
    output_root.mkdir(parents=True, exist_ok=True)

    reference_data_dir = (
        Path(args.reference_data_dir).expanduser().resolve()
        if args.reference_data_dir is not None
        else None
    )
    cache_dir = None
    if "cache_dir" in cfg.datasets[args.part]:
        cache_dir = Path(str(cfg.datasets[args.part].cache_dir)).expanduser().resolve()

    model = instantiate(cfg.model).to(device)
    model.eval()

    roi_size = tuple(int(x) for x in cfg.trainer.get("sw_roi_size", [96, 96, 96]))
    sw_batch_size = int(cfg.trainer.get("sw_batch_size", 1))
    overlap = float(cfg.trainer.get("sw_overlap", 0.5))

    global_summary = {
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "part": args.part,
        "tta_enabled": bool(tta_enabled),
        "tta_views": 8 if tta_enabled else 1,
        "roi_size": list(roi_size),
        "sw_batch_size": sw_batch_size,
        "sw_overlap": overlap,
        "threshold": float(args.threshold),
        "write_pt": bool(write_pt),
        "write_nifti": bool(write_nifti),
        "checkpoints": [],
    }

    for ckpt_info in checkpoints:
        checkpoint_path = Path(ckpt_info["checkpoint"]).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (run_dir / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            print(f"[WARN] checkpoint not found, skip: {checkpoint_path}")
            continue

        alias = checkpoint_alias(checkpoint_path)
        model_out_dir = output_root / alias
        if write_pt:
            (model_out_dir / "pt").mkdir(parents=True, exist_ok=True)
        if write_nifti:
            (model_out_dir / "nifti").mkdir(parents=True, exist_ok=True)

        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        has_label = False
        dice_acc = init_dice_accumulator(device=device)
        case_records = []

        for batch in tqdm(
            dataloader,
            desc=f"{alias}:{args.part}",
            total=len(dataloader),
        ):
            image = batch["image"].to(device)
            logits = sliding_window_with_tta(
                model=model,
                image=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                tta_enabled=bool(tta_enabled),
            )

            probs = torch.sigmoid(logits)
            pred_channels = (probs > float(args.threshold)).float()

            label = batch.get("label")
            if label is not None:
                has_label = True
                label = label.to(device).float()
                update_dice_accumulator(
                    acc=dice_acc,
                    pred_channels=pred_channels,
                    label_channels=label,
                )

            case_ids = batch.get("case_id")
            batch_size = pred_channels.shape[0]
            if case_ids is None:
                case_ids = [f"case_{idx}" for idx in range(batch_size)]

            for idx in range(batch_size):
                case_id = str(case_ids[idx])
                seg_crop = channels_to_brats_label(pred_channels[idx])
                meta = load_case_meta(case_id, vector_path_map, meta_cache)
                seg = restore_to_orig_shape(seg_crop, meta)

                if write_pt:
                    torch.save(
                        {
                            "case_id": case_id,
                            "segmentation": seg.cpu(),
                            "meta": meta,
                        },
                        model_out_dir / "pt" / f"{case_id}.pt",
                    )

                nifti_saved = False
                if write_nifti:
                    reference_nii = find_reference_nii(
                        case_id=case_id,
                        reference_data_dir=reference_data_dir,
                        cache_dir=cache_dir,
                    )
                    if reference_nii is not None:
                        save_nifti(
                            seg=seg,
                            output_path=model_out_dir / "nifti" / f"{case_id}-seg.nii.gz",
                            reference_path=reference_nii,
                        )
                        nifti_saved = True

                case_records.append(
                    {
                        "case_id": case_id,
                        "pt_saved": bool(write_pt),
                        "nifti_saved": bool(nifti_saved),
                    }
                )

        model_summary = {
            "checkpoint": str(checkpoint_path),
            "source": ckpt_info.get("source"),
            "score": ckpt_info.get("score"),
            "metric": ckpt_info.get("metric"),
            "num_cases": len(case_records),
            "has_label": bool(has_label),
            "cases": case_records,
        }
        if has_label:
            model_summary["metrics"] = finalize_dice_accumulator(dice_acc)

        with (model_out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(model_summary, f, indent=2, ensure_ascii=False)

        global_summary["checkpoints"].append(model_summary)

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print(f"Done. Results saved to: {output_root}")


if __name__ == "__main__":
    main()
