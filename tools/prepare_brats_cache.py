#!/usr/bin/env python3
"""
Prepare BraTS23 cache under data/cached with two artifacts:
1) Optional uncompressed NIfTI files (.nii) for faster repeated reads.
2) Vector cache files (.pt) used directly by training dataloaders.

Vector file schema:
{
    "image": FloatTensor [4, D, H, W], normalized channel-wise on nonzero voxels,
    "label": FloatTensor [3, D, H, W], channels = [TC, WT, ET],
    "case_id": str,
}
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

BRATS_MODALITIES = ("t2f", "t1c", "t1n", "t2w")


def normalize_nonzero_channelwise(image: np.ndarray) -> np.ndarray:
    """
    Normalize each channel using nonzero voxels only.
    Input shape: [C, D, H, W]
    """
    output = image.astype(np.float32, copy=True)
    for channel_idx in range(output.shape[0]):
        channel = output[channel_idx]
        mask = channel != 0
        if not np.any(mask):
            continue
        values = channel[mask]
        mean = values.mean()
        std = values.std()
        if std < 1e-6:
            std = 1.0
        output[channel_idx] = (channel - mean) / std
    return output


def label_to_brats3_channels(label: np.ndarray) -> np.ndarray:
    """
    Convert label map to 3 channels [TC, WT, ET] for BraTS23 convention.
    - TC: label 1 or 3
    - WT: label 1 or 2 or 3
    - ET: label 3
    """
    tc = (label == 1) | (label == 3)
    wt = (label == 1) | (label == 2) | (label == 3)
    et = label == 3
    return np.stack([tc, wt, et], axis=0).astype(np.float32)


def maybe_write_uncompressed_nifti(src_nii_gz: Path, dst_nii: Path) -> None:
    if dst_nii.exists():
        return
    nii_obj = nib.load(str(src_nii_gz))
    dst_nii.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii_obj, str(dst_nii))


def build_case_paths(case_dir: Path):
    case_id = case_dir.name
    image_paths = [case_dir / f"{case_id}-{mod}.nii.gz" for mod in BRATS_MODALITIES]
    label_path = case_dir / f"{case_id}-seg.nii.gz"
    if not all(path.exists() for path in image_paths) or not label_path.exists():
        return None
    return case_id, image_paths, label_path


def prepare_cache(
    data_dir: Path,
    cache_dir: Path,
    write_uncompressed_nii: bool,
    overwrite_vectors: bool,
) -> None:
    data_dir = data_dir.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    cases_root = cache_dir / "vectors"
    nii_root = cache_dir / "nifti"
    index_path = cache_dir / "index.json"

    case_dirs = sorted(path for path in data_dir.glob("BraTS-GLI-*") if path.is_dir())
    if not case_dirs:
        raise ValueError(f"No BraTS-GLI-* case directories found under: {data_dir}")

    cases_root.mkdir(parents=True, exist_ok=True)
    if write_uncompressed_nii:
        nii_root.mkdir(parents=True, exist_ok=True)

    index = []
    for case_dir in tqdm(case_dirs, desc="Preparing cache"):
        built = build_case_paths(case_dir)
        if built is None:
            continue
        case_id, image_paths, label_path = built

        vector_path = cases_root / f"{case_id}.pt"
        if vector_path.exists() and not overwrite_vectors:
            index.append({"case_id": case_id, "vector_path": str(vector_path)})
            continue

        if write_uncompressed_nii:
            for mod, src_path in zip(BRATS_MODALITIES, image_paths):
                maybe_write_uncompressed_nifti(
                    src_nii_gz=src_path,
                    dst_nii=nii_root / case_id / f"{case_id}-{mod}.nii",
                )
            maybe_write_uncompressed_nifti(
                src_nii_gz=label_path,
                dst_nii=nii_root / case_id / f"{case_id}-seg.nii",
            )

        image_arrays = [
            nib.load(str(path)).get_fdata(dtype=np.float32)
            for path in image_paths
        ]
        image = np.stack(image_arrays, axis=0)  # [4, D, H, W]
        image = normalize_nonzero_channelwise(image=image)

        label = nib.load(str(label_path)).get_fdata(dtype=np.float32)
        label = label_to_brats3_channels(label=label)

        payload = {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "case_id": case_id,
        }
        torch.save(payload, vector_path)

        index.append({"case_id": case_id, "vector_path": str(vector_path)})

    cache_dir.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Done. Cached vectors: {len(index)}")
    print(f"Index file: {index_path}")
    print(f"Vectors dir: {cases_root}")
    if write_uncompressed_nii:
        print(f"Uncompressed NIfTI dir: {nii_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare BraTS23 cached vectors.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="BraTS23 source directory containing BraTS-GLI-* case folders.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cached/brats23"),
        help="Output cache directory (default: data/cached/brats23).",
    )
    parser.add_argument(
        "--no-uncompressed-nii",
        action="store_true",
        help="Disable writing uncompressed .nii files.",
    )
    parser.add_argument(
        "--overwrite-vectors",
        action="store_true",
        help="Recreate existing .pt vectors if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_cache(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        write_uncompressed_nii=not args.no_uncompressed_nii,
        overwrite_vectors=args.overwrite_vectors,
    )


if __name__ == "__main__":
    main()
