#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.brats23 import select_subset, split_index_kfold, split_index_three_way

try:
    import numpy as np
except Exception:
    np = None


def _read_cached_index(cache_dir: Path) -> list[dict]:
    index_path = cache_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Cached index file not found: {index_path}. "
            "Run tools/prepare_brats_cache.py first."
        )
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, list) or len(index) == 0:
        raise ValueError(f"Cached index is empty or invalid: {index_path}")
    return index


def _resolve_vector_path(cache_dir: Path, record: dict) -> Path:
    vector_root = cache_dir / "vectors"
    candidates = []
    indexed_path = record.get("vector_path")
    if isinstance(indexed_path, str) and indexed_path.strip():
        raw_path = Path(indexed_path).expanduser()
        if raw_path.is_absolute():
            candidates.append(raw_path.resolve())
        else:
            candidates.append((cache_dir / raw_path).resolve())
            candidates.append(raw_path.resolve())
        candidates.append((vector_root / raw_path.name).resolve())

    case_id = record.get("case_id")
    if isinstance(case_id, str) and case_id:
        candidates.append((vector_root / f"{case_id}.pt").resolve())

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Cached vector file not found.\n"
        f"case_id={record.get('case_id', '<unknown>')}\n"
        f"indexed vector_path={indexed_path}\n"
        f"cache_dir={cache_dir}\n"
        f"tried paths:\n{tried}"
    )


def _torch_load_compat(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_scalar_label(label: torch.Tensor) -> torch.Tensor:
    label = torch.as_tensor(label)
    if label.ndim == 3:
        return label.long()
    if label.ndim == 4 and label.shape[0] == 1:
        return label[0].long()
    if label.ndim == 4 and label.shape[0] == 3:
        tc = label[0] > 0.5
        wt = label[1] > 0.5
        et = label[2] > 0.5
        scalar = torch.zeros_like(label[0], dtype=torch.long)
        scalar[wt] = 2
        scalar[tc] = 1
        scalar[et] = 3
        return scalar
    raise ValueError(f"Unsupported label shape: {tuple(label.shape)}")


def _iter_connected_components(mask: torch.Tensor, connectivity: int):
    mask = torch.as_tensor(mask).bool().cpu()
    if not bool(mask.any().item()):
        return

    if np is None:
        # Fallback without numpy/scipy: keep one full ET region.
        yield mask
        return

    try:
        from scipy import ndimage as ndi

        rank = 1 if int(connectivity) == 6 else 3
        structure = ndi.generate_binary_structure(3, rank)
        mask_np = mask.numpy().astype(np.uint8)
        labeled, n_labels = ndi.label(mask_np, structure=structure)
        for label_id in range(1, int(n_labels) + 1):
            component = torch.from_numpy(labeled == label_id).bool()
            if bool(component.any().item()):
                yield component
        return
    except Exception:
        pass

    # Fallback without scipy: keep one full ET region.
    yield mask


def _bbox_from_mask(mask: torch.Tensor):
    coords = torch.nonzero(torch.as_tensor(mask).bool(), as_tuple=False)
    if int(coords.numel()) == 0:
        return None
    mins = [int(coords[:, i].min().item()) for i in range(3)]
    maxs = [int(coords[:, i].max().item()) + 1 for i in range(3)]
    return mins, maxs


def _expand_bbox(mins, maxs, spatial_shape, margin: int):
    out_mins = []
    out_maxs = []
    for i in range(3):
        lo = max(0, int(mins[i]) - int(margin))
        hi = min(int(spatial_shape[i]), int(maxs[i]) + int(margin))
        out_mins.append(lo)
        out_maxs.append(hi)
    return out_mins, out_maxs


def _select_partition_index(
    all_index: list[dict],
    partition: str,
    usage_ratio: float,
    split_strategy: str,
    val_ratio: float,
    test_ratio: float,
    fold: int,
    n_folds: int,
    seed: int,
) -> list[dict]:
    selected = select_subset(index=all_index, usage_ratio=usage_ratio, seed=seed)
    if partition == "all":
        return selected

    if split_strategy == "three_way":
        train_index, val_index, test_index = split_index_three_way(
            index=selected,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        if partition == "train":
            return train_index
        if partition == "val":
            return val_index
        return test_index

    if split_strategy == "kfold":
        train_index, val_index = split_index_kfold(
            index=selected,
            fold=fold,
            n_folds=n_folds,
        )
        if partition == "train":
            return train_index
        return val_index

    raise ValueError(f"Unsupported split_strategy: {split_strategy}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ET copy-paste donor bank from cached BraTS vectors."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Cached vectors root containing index.json and vectors/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output donor bank .pt path.",
    )
    parser.add_argument(
        "--partition",
        choices=["train", "val", "test", "all"],
        default="train",
        help="Which split to extract donors from.",
    )
    parser.add_argument(
        "--usage-ratio",
        type=float,
        default=1.0,
        help="Subset usage ratio in (0,1].",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["three_way", "kfold"],
        default="three_way",
        help="Split strategy consistent with training config.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--context-margin",
        type=int,
        default=4,
        help="Extra voxels around ET component bbox.",
    )
    parser.add_argument(
        "--min-et-voxels",
        type=int,
        default=8,
        help="Minimum ET voxels per donor component.",
    )
    parser.add_argument(
        "--max-et-voxels",
        type=int,
        default=2000,
        help="Maximum ET voxels per donor component (<=0 to disable).",
    )
    parser.add_argument(
        "--max-donors",
        type=int,
        default=300,
        help="Keep smallest N donors by ET voxel count (<=0 means keep all).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[6, 26],
        default=26,
        help="Connected component neighborhood.",
    )
    parser.add_argument(
        "--image-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Saved donor image dtype.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    all_index = _read_cached_index(cache_dir=cache_dir)
    part_index = _select_partition_index(
        all_index=all_index,
        partition=args.partition,
        usage_ratio=float(args.usage_ratio),
        split_strategy=args.split_strategy,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        fold=int(args.fold),
        n_folds=int(args.n_folds),
        seed=int(args.seed),
    )

    donors = []
    case_count = 0
    for record in tqdm(part_index, desc="Building ET donors"):
        vector_path = _resolve_vector_path(cache_dir=cache_dir, record=record)
        payload = _torch_load_compat(vector_path)
        image = torch.as_tensor(payload["image"]).float()
        scalar = _to_scalar_label(payload["label"])
        et_mask = (scalar == 3).bool().cpu()
        if not bool(et_mask.any().item()):
            continue

        case_count += 1
        for component in _iter_connected_components(
            et_mask, connectivity=int(args.connectivity)
        ):
            et_voxels = int(component.sum().item())
            if et_voxels < int(args.min_et_voxels):
                continue
            if int(args.max_et_voxels) > 0 and et_voxels > int(args.max_et_voxels):
                continue

            bbox = _bbox_from_mask(component)
            if bbox is None:
                continue
            mins, maxs = _expand_bbox(
                mins=bbox[0],
                maxs=bbox[1],
                spatial_shape=tuple(int(x) for x in et_mask.shape),
                margin=int(args.context_margin),
            )
            z0, y0, x0 = mins
            z1, y1, x1 = maxs
            image_crop = image[:, z0:z1, y0:y1, x0:x1]
            mask_crop = component[z0:z1, y0:y1, x0:x1].to(torch.uint8)

            if args.image_dtype == "float16":
                image_crop = image_crop.to(torch.float16)
            else:
                image_crop = image_crop.to(torch.float32)

            donors.append(
                {
                    "case_id": str(record.get("case_id", vector_path.stem)),
                    "image": image_crop.contiguous(),
                    "mask_et": mask_crop.contiguous(),
                    "et_voxels": int(et_voxels),
                    "bbox_size": [int(z1 - z0), int(y1 - y0), int(x1 - x0)],
                }
            )

    if len(donors) == 0:
        raise RuntimeError(
            "No ET donors were extracted. "
            "Try lowering --min-et-voxels or increasing --max-et-voxels."
        )

    donors.sort(key=lambda item: int(item["et_voxels"]))
    if int(args.max_donors) > 0:
        donors = donors[: int(args.max_donors)]

    volumes = torch.tensor([int(item["et_voxels"]) for item in donors], dtype=torch.float32)
    payload = {
        "version": 1,
        "source_cache_dir": str(cache_dir),
        "partition": args.partition,
        "num_cases_with_et": int(case_count),
        "num_donors": int(len(donors)),
        "min_et_voxels": int(volumes.min().item()),
        "median_et_voxels": float(volumes.median().item()),
        "max_et_voxels": int(volumes.max().item()),
        "donors": donors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved ET donor bank to: {output_path}")
    print(
        "Summary: "
        f"donors={payload['num_donors']} "
        f"cases_with_et={payload['num_cases_with_et']} "
        f"et_voxels[min/median/max]="
        f"{payload['min_et_voxels']}/{payload['median_et_voxels']:.1f}/{payload['max_et_voxels']}"
    )


if __name__ == "__main__":
    main()
