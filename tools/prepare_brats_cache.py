#!/usr/bin/env python3
"""
Prepare BraTS23 cache under data/cached with two artifacts:
1) Optional uncompressed NIfTI files (.nii) for faster repeated reads.
2) Vector cache files (.pt) used directly by training dataloaders.

Optimized vector schema (smaller disk footprint for HDD):
{
    "image": FloatTensor [4, D, H, W], usually float16 after normalization,
    "label": UInt8Tensor [1, D, H, W], scalar class map in {0,1,2,3},
    "case_id": str,
    "meta": {
        "orig_shape": [D, H, W],
        "cropped_shape": [D, H, W],
        "bbox_start": [d0, h0, w0],
        "bbox_end": [d1, h1, w1],
    },
}
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover - optional dependency for N4 correction
    sitk = None

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


def _foreground_bbox(image: np.ndarray):
    """
    Compute foreground bbox from multi-modal image.
    Input shape: [C, D, H, W]
    Returns:
        ((d0, d1), (h0, h1), (w0, w1)) with end-exclusive indices.
    """
    fg = np.any(image != 0, axis=0)
    coords = np.where(fg)
    if coords[0].size == 0:
        _, d, h, w = image.shape
        return (0, d), (0, h), (0, w)

    d0, d1 = int(coords[0].min()), int(coords[0].max()) + 1
    h0, h1 = int(coords[1].min()), int(coords[1].max()) + 1
    w0, w1 = int(coords[2].min()), int(coords[2].max()) + 1
    return (d0, d1), (h0, h1), (w0, w1)


def _align_interval(start, end, limit, margin, min_size, k_divisible):
    start = max(0, int(start) - margin)
    end = min(limit, int(end) + margin)
    if end <= start:
        return 0, limit

    target = max(end - start, int(min_size))
    if k_divisible > 1:
        target = ((target + k_divisible - 1) // k_divisible) * k_divisible
    if target >= limit:
        return 0, limit

    center = (start + end) // 2
    new_start = center - target // 2
    new_end = new_start + target

    if new_start < 0:
        new_end -= new_start
        new_start = 0
    if new_end > limit:
        shift = new_end - limit
        new_start -= shift
        new_end = limit
        if new_start < 0:
            new_start = 0

    return int(new_start), int(new_end)


def crop_to_foreground(
    image: np.ndarray,
    label: np.ndarray,
    margin: int,
    min_size: tuple[int, int, int],
    k_divisible: int,
):
    """
    Crop image/label to foreground bbox, then align shape by constraints.
    """
    (d0, d1), (h0, h1), (w0, w1) = _foreground_bbox(image)
    _, d_lim, h_lim, w_lim = image.shape

    d0, d1 = _align_interval(
        d0, d1, d_lim, margin=margin, min_size=min_size[0], k_divisible=k_divisible
    )
    h0, h1 = _align_interval(
        h0, h1, h_lim, margin=margin, min_size=min_size[1], k_divisible=k_divisible
    )
    w0, w1 = _align_interval(
        w0, w1, w_lim, margin=margin, min_size=min_size[2], k_divisible=k_divisible
    )

    image_cropped = image[:, d0:d1, h0:h1, w0:w1]
    label_cropped = label[d0:d1, h0:h1, w0:w1]
    return image_cropped, label_cropped, (d0, h0, w0), (d1, h1, w1)


def maybe_write_uncompressed_nifti(src_nii_gz: Path, dst_nii: Path) -> None:
    if dst_nii.exists():
        return
    nii_obj = nib.load(str(src_nii_gz))
    dst_nii.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii_obj, str(dst_nii))


def load_nifti_array_with_optional_n4(
    nii_path: Path,
    enable_n4: bool,
    n4_shrink_factor: int,
    n4_iterations: tuple[int, ...],
) -> np.ndarray:
    """
    Load a scalar NIfTI volume as float32 and optionally apply N4 bias correction.

    N4 is applied in array space:
    - Convert [X, Y, Z] -> [Z, Y, X] for SimpleITK.
    - Build Otsu foreground mask.
    - Run N4 on a shrunk image for speed, then recover full-resolution bias field.
    - Convert back to [X, Y, Z].
    """
    image = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
    if not enable_n4:
        return image

    if sitk is None:
        raise ImportError(
            "N4 bias correction requires SimpleITK. "
            "Please install dependency 'simpleitk' and rerun."
        )

    if n4_shrink_factor < 1:
        raise ValueError(f"n4_shrink_factor must be >= 1, got {n4_shrink_factor}")
    if len(n4_iterations) == 0:
        raise ValueError("n4_iterations must contain at least one level.")

    image_zyx = np.transpose(image, (2, 1, 0))
    sitk_image = sitk.GetImageFromArray(image_zyx)
    mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(x) for x in n4_iterations])

    if n4_shrink_factor > 1:
        shrink = [int(n4_shrink_factor)] * 3
        image_small = sitk.Shrink(sitk_image, shrink)
        mask_small = sitk.Shrink(mask, shrink)
        corrector.Execute(image_small, mask_small)
        log_bias_full = corrector.GetLogBiasFieldAsImage(sitk_image)
        corrected = sitk_image / sitk.Exp(log_bias_full)
    else:
        corrected = corrector.Execute(sitk_image, mask)

    corrected_zyx = sitk.GetArrayFromImage(corrected).astype(np.float32, copy=False)
    return np.transpose(corrected_zyx, (2, 1, 0))


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
    crop_foreground: bool,
    crop_margin: int,
    min_size: tuple[int, int, int],
    k_divisible: int,
    image_dtype: str,
    n4_correct: bool,
    n4_shrink_factor: int,
    n4_iterations: tuple[int, ...],
    num_workers: int,
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
    if num_workers <= 1:
        for case_dir in tqdm(case_dirs, desc="Preparing cache"):
            item = _prepare_case_worker(
                case_dir=case_dir,
                cases_root=cases_root,
                nii_root=nii_root,
                write_uncompressed_nii=write_uncompressed_nii,
                overwrite_vectors=overwrite_vectors,
                crop_foreground=crop_foreground,
                crop_margin=crop_margin,
                min_size=min_size,
                k_divisible=k_divisible,
                image_dtype=image_dtype,
                n4_correct=n4_correct,
                n4_shrink_factor=n4_shrink_factor,
                n4_iterations=n4_iterations,
            )
            if item is not None:
                index.append(item)
    else:
        worker_kwargs = {
            "cases_root": cases_root,
            "nii_root": nii_root,
            "write_uncompressed_nii": write_uncompressed_nii,
            "overwrite_vectors": overwrite_vectors,
            "crop_foreground": crop_foreground,
            "crop_margin": crop_margin,
            "min_size": min_size,
            "k_divisible": k_divisible,
            "image_dtype": image_dtype,
            "n4_correct": n4_correct,
            "n4_shrink_factor": n4_shrink_factor,
            "n4_iterations": n4_iterations,
        }
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_prepare_case_worker, case_dir=case_dir, **worker_kwargs): case_dir
                for case_dir in case_dirs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing cache"):
                case_dir = futures[future]
                try:
                    item = future.result()
                except Exception as error:
                    raise RuntimeError(f"Failed to prepare case: {case_dir}") from error
                if item is not None:
                    index.append(item)

    index.sort(key=lambda x: x["case_id"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Done. Cached vectors: {len(index)}")
    print(f"Index file: {index_path}")
    print(f"Vectors dir: {cases_root}")
    if write_uncompressed_nii:
        print(f"Uncompressed NIfTI dir: {nii_root}")


def _prepare_case_worker(
    case_dir: Path,
    cases_root: Path,
    nii_root: Path,
    write_uncompressed_nii: bool,
    overwrite_vectors: bool,
    crop_foreground: bool,
    crop_margin: int,
    min_size: tuple[int, int, int],
    k_divisible: int,
    image_dtype: str,
    n4_correct: bool,
    n4_shrink_factor: int,
    n4_iterations: tuple[int, ...],
):
    built = build_case_paths(case_dir)
    if built is None:
        return None
    case_id, image_paths, label_path = built

    vector_path = cases_root / f"{case_id}.pt"
    if vector_path.exists() and not overwrite_vectors:
        return {"case_id": case_id, "vector_path": str(vector_path)}

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
        load_nifti_array_with_optional_n4(
            nii_path=path,
            enable_n4=n4_correct,
            n4_shrink_factor=n4_shrink_factor,
            n4_iterations=n4_iterations,
        )
        for path in image_paths
    ]
    image = np.stack(image_arrays, axis=0)  # [4, D, H, W]
    image = normalize_nonzero_channelwise(image=image)

    label = nib.load(str(label_path)).get_fdata(dtype=np.float32).astype(np.uint8)
    label[label == 4] = 3  # BraTS2021 compatibility

    orig_shape = tuple(int(x) for x in image.shape[1:])
    bbox_start = (0, 0, 0)
    bbox_end = orig_shape
    if crop_foreground:
        image, label, bbox_start, bbox_end = crop_to_foreground(
            image=image,
            label=label,
            margin=crop_margin,
            min_size=min_size,
            k_divisible=k_divisible,
        )

    if image_dtype == "float16":
        image = image.astype(np.float16, copy=False)
    else:
        image = image.astype(np.float32, copy=False)
    label = label[np.newaxis, ...].astype(np.uint8, copy=False)  # [1, D, H, W]

    payload = {
        "image": torch.from_numpy(image),
        "label": torch.from_numpy(label),
        "case_id": case_id,
        "meta": {
            "orig_shape": list(orig_shape),
            "cropped_shape": [int(x) for x in image.shape[1:]],
            "bbox_start": [int(x) for x in bbox_start],
            "bbox_end": [int(x) for x in bbox_end],
            "n4_corrected": bool(n4_correct),
            "n4_shrink_factor": int(n4_shrink_factor),
            "n4_iterations": [int(x) for x in n4_iterations],
        },
    }
    torch.save(payload, vector_path)
    return {"case_id": case_id, "vector_path": str(vector_path)}


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
    parser.add_argument(
        "--no-crop-foreground",
        action="store_true",
        help="Disable foreground crop for cached vectors.",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=4,
        help="Foreground bbox expansion margin in voxels.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        metavar=("D", "H", "W"),
        help="Minimum cropped spatial size before saving vectors.",
    )
    parser.add_argument(
        "--k-divisible",
        type=int,
        default=32,
        help="Align cropped shape to be divisible by this value.",
    )
    parser.add_argument(
        "--image-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Image tensor dtype stored in vectors.",
    )
    parser.add_argument(
        "--n4-correct",
        action="store_true",
        help="Apply N4 bias field correction to each MRI modality before normalization.",
    )
    parser.add_argument(
        "--n4-shrink-factor",
        type=int,
        default=2,
        help="N4 shrink factor for faster correction (1 means full resolution).",
    )
    parser.add_argument(
        "--n4-iterations",
        type=int,
        nargs="+",
        default=[50, 50, 30, 20],
        help="N4 max iterations per level, e.g. 50 50 30 20.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for cache preparation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_cache(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        write_uncompressed_nii=not args.no_uncompressed_nii,
        overwrite_vectors=args.overwrite_vectors,
        crop_foreground=not args.no_crop_foreground,
        crop_margin=args.crop_margin,
        min_size=tuple(args.min_size),
        k_divisible=args.k_divisible,
        image_dtype=args.image_dtype,
        n4_correct=args.n4_correct,
        n4_shrink_factor=args.n4_shrink_factor,
        n4_iterations=tuple(args.n4_iterations),
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
