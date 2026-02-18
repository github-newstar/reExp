import random
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

BRATS23_MODALITIES = ("t2f", "t1c", "t1n", "t2w")


def build_brats23_index(data_dir, require_label=True):
    """
    Build sample index from BraTS23-like folder structure.

    Expected case folder format:
    - BraTS-GLI-xxxxx-xxx/
      - BraTS-GLI-xxxxx-xxx-t2f.nii.gz
      - BraTS-GLI-xxxxx-xxx-t1c.nii.gz
      - BraTS-GLI-xxxxx-xxx-t1n.nii.gz
      - BraTS-GLI-xxxxx-xxx-t2w.nii.gz
      - BraTS-GLI-xxxxx-xxx-seg.nii.gz
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"BraTS23 data_dir does not exist: {root}")

    case_dirs = sorted(path for path in root.glob("BraTS-GLI-*") if path.is_dir())
    if not case_dirs:
        raise ValueError(f"No BraTS-GLI-* case directories found under: {root}")

    index = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        image_paths = [case_dir / f"{case_id}-{mod}.nii.gz" for mod in BRATS23_MODALITIES]
        if not all(path.exists() for path in image_paths):
            continue

        sample = {
            "image": [str(path) for path in image_paths],
            "case_id": case_id,
        }

        label_path = case_dir / f"{case_id}-seg.nii.gz"
        if require_label:
            if not label_path.exists():
                continue
            sample["label"] = str(label_path)
        elif label_path.exists():
            sample["label"] = str(label_path)

        index.append(sample)

    if not index:
        raise ValueError(
            "No valid BraTS23 samples found. "
            "Check modality files and labels under the provided data_dir."
        )

    return index


def select_subset(index, usage_ratio, seed):
    """
    Select a random subset for quick experiments while preserving reproducibility.
    """
    if not (0 < usage_ratio <= 1):
        raise ValueError(f"usage_ratio must be in (0, 1], got {usage_ratio}")

    selected = list(index)
    rng = random.Random(seed)
    rng.shuffle(selected)

    n_selected = max(1, int(len(selected) * usage_ratio))
    return selected[:n_selected]


def split_index_kfold(index, fold, n_folds):
    """
    Split index into train and validation folds.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0, {n_folds - 1}], got {fold}")

    fold_size = len(index) // n_folds
    if fold_size == 0:
        raise ValueError(
            f"Not enough samples ({len(index)}) for n_folds={n_folds}. "
            "Decrease n_folds or increase usage_ratio."
        )

    start_idx = fold * fold_size
    end_idx = len(index) if fold == n_folds - 1 else start_idx + fold_size

    val_index = index[start_idx:end_idx]
    train_index = index[:start_idx] + index[end_idx:]
    return train_index, val_index


class BraTS23Dataset(Dataset):
    """
    BraTS23 dataset adapter following template expectations.

    Returned sample keys follow segmentation conventions:
    - image: multi-modal volume
    - label: segmentation target (if available)
    - case_id: case identifier
    """

    def __init__(
        self,
        data_dir,
        partition,
        usage_ratio=1.0,
        fold=0,
        n_folds=5,
        seed=42,
        require_label=True,
        instance_transforms=None,
    ):
        """
        Args:
            data_dir (str): BraTS23 root directory.
            partition (str): one of {"train", "val", "test", "all"}.
            usage_ratio (float): dataset usage ratio in (0, 1].
            fold (int): validation fold id.
            n_folds (int): number of folds.
            seed (int): random seed for subset shuffle.
            require_label (bool): if True, skip samples without labels.
            instance_transforms (Callable | None): transform pipeline applied
                to each sample dict.
        """
        if partition not in {"train", "val", "test", "all"}:
            raise ValueError(
                f"partition must be one of {{'train', 'val', 'test', 'all'}}, got {partition}"
            )

        all_index = build_brats23_index(data_dir=data_dir, require_label=require_label)
        selected_index = select_subset(index=all_index, usage_ratio=usage_ratio, seed=seed)

        if partition == "all":
            self._index = selected_index
        else:
            train_index, val_index = split_index_kfold(
                index=selected_index,
                fold=fold,
                n_folds=n_folds,
            )
            if partition == "train":
                self._index = train_index
            else:
                self._index = val_index

        self.instance_transforms = instance_transforms

    def __len__(self):
        return len(self._index)

    def __getitem__(self, ind):
        sample = deepcopy(self._index[ind])
        if self.instance_transforms is not None:
            sample = self.instance_transforms(sample)
        return sample


def _read_cached_index(cache_dir):
    index_path = Path(cache_dir).expanduser().resolve() / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Cached index file not found: {index_path}. "
            "Run tools/prepare_brats_cache.py first."
        )
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f, object_pairs_hook=OrderedDict)
    if not isinstance(index, list) or len(index) == 0:
        raise ValueError(f"Cached index is empty or invalid: {index_path}")
    return index


class BraTS23CachedVectorDataset(Dataset):
    """
    Dataset that loads preprocessed tensor vectors (.pt) from disk cache.

    Expected each cache item to contain keys:
    - image: Tensor [4, D, H, W]
    - label: Tensor [3, D, H, W]
    - case_id: str
    """

    def __init__(
        self,
        cache_dir,
        partition,
        usage_ratio=1.0,
        fold=0,
        n_folds=5,
        seed=42,
        instance_transforms=None,
    ):
        if partition not in {"train", "val", "test", "all"}:
            raise ValueError(
                f"partition must be one of {{'train', 'val', 'test', 'all'}}, got {partition}"
            )

        all_index = _read_cached_index(cache_dir=cache_dir)
        selected_index = select_subset(index=all_index, usage_ratio=usage_ratio, seed=seed)

        if partition == "all":
            self._index = selected_index
        else:
            train_index, val_index = split_index_kfold(
                index=selected_index,
                fold=fold,
                n_folds=n_folds,
            )
            self._index = train_index if partition == "train" else val_index

        self.instance_transforms = instance_transforms

    def __len__(self):
        return len(self._index)

    def __getitem__(self, ind):
        record = self._index[ind]
        vector_path = Path(record["vector_path"]).expanduser().resolve()
        sample = torch.load(vector_path, map_location="cpu")
        if self.instance_transforms is not None:
            sample = self.instance_transforms(sample)
        return sample
