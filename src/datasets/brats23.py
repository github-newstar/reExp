import random
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from tqdm.auto import tqdm

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


def split_index_three_way(index, val_ratio, test_ratio):
    """
    Split index into mutually exclusive train/val/test subsets.

    The input index is expected to be pre-shuffled (we already do this in
    `select_subset` with a fixed seed), so a simple contiguous split is
    deterministic and reproducible.
    """
    if not (0 <= val_ratio < 1):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not (0 <= test_ratio < 1):
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if val_ratio + test_ratio >= 1:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio}"
        )

    n_total = len(index)
    if n_total < 3:
        raise ValueError(
            f"Need at least 3 samples for independent train/val/test split, got {n_total}"
        )

    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    if test_ratio > 0 and n_test == 0:
        n_test = 1
    if val_ratio > 0 and n_val == 0:
        n_val = 1

    # Keep at least one training sample.
    while n_test + n_val >= n_total:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    test_index = index[:n_test]
    val_index = index[n_test : n_test + n_val]
    train_index = index[n_test + n_val :]
    return train_index, val_index, test_index


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
        split_strategy="three_way",
        val_ratio=0.1,
        test_ratio=0.1,
        fold=0,
        n_folds=5,
        seed=42,
        require_label=True,
        instance_transforms=None,
        cache_in_memory=False,
    ):
        """
        Args:
            data_dir (str): BraTS23 root directory.
            partition (str): one of {"train", "val", "test", "all"}.
            usage_ratio (float): dataset usage ratio in (0, 1].
            split_strategy (str): one of {"three_way", "kfold"}.
            val_ratio (float): validation ratio for three-way split.
            test_ratio (float): test ratio for three-way split.
            fold (int): validation fold id.
            n_folds (int): number of folds.
            seed (int): random seed for subset shuffle.
            require_label (bool): if True, skip samples without labels.
            instance_transforms (Callable | None): transform pipeline applied
                to each sample dict.
            cache_in_memory (bool): if True, preload transformed samples
                into RAM at startup (recommended for val/test only).
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
            if split_strategy == "three_way":
                train_index, val_index, test_index = split_index_three_way(
                    index=selected_index,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                )
                if partition == "train":
                    self._index = train_index
                elif partition == "val":
                    self._index = val_index
                else:
                    self._index = test_index
            elif split_strategy == "kfold":
                train_index, val_index = split_index_kfold(
                    index=selected_index,
                    fold=fold,
                    n_folds=n_folds,
                )
                if partition == "train":
                    self._index = train_index
                else:
                    # Keep backward compatibility: val/test both use val fold.
                    self._index = val_index
            else:
                raise ValueError(
                    f"split_strategy must be one of {{'three_way', 'kfold'}}, got {split_strategy}"
                )

        self.instance_transforms = instance_transforms
        self.cache_in_memory = bool(cache_in_memory)
        self._memory_cache = None
        if self.cache_in_memory:
            num_workers = max(1, min(mp.cpu_count() - 1, 32))
            
            # Use file_system strategy to avoid "received 0 items of ancdata" (file descriptor limit)
            mp.set_sharing_strategy('file_system')
            with mp.Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._load_and_transform_for_mp, range(len(self._index))),
                        total=len(self._index),
                        desc=f"Caching {partition} in memory (MP, workers={num_workers})",
                    )
                )
            self._memory_cache = results

    def _load_and_transform_for_mp(self, ind):
        return self._load_and_transform(ind)

    def __len__(self):
        return len(self._memory_cache) if self._memory_cache is not None else len(self._index)

    def _load_and_transform(self, ind):
        sample = deepcopy(self._index[ind])
        if self.instance_transforms is not None:
            sample = self.instance_transforms(sample)
        return sample

    def __getitem__(self, ind):
        if self._memory_cache is not None:
            # Return a copy so per-batch device moves do not mutate cached items.
            return deepcopy(self._memory_cache[ind])
        return self._load_and_transform(ind)


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
        split_strategy="three_way",
        val_ratio=0.1,
        test_ratio=0.1,
        fold=0,
        n_folds=5,
        seed=42,
        instance_transforms=None,
        use_mmap=True,
        cache_in_memory=False,
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
            if split_strategy == "three_way":
                train_index, val_index, test_index = split_index_three_way(
                    index=selected_index,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                )
                if partition == "train":
                    self._index = train_index
                elif partition == "val":
                    self._index = val_index
                else:
                    self._index = test_index
            elif split_strategy == "kfold":
                train_index, val_index = split_index_kfold(
                    index=selected_index,
                    fold=fold,
                    n_folds=n_folds,
                )
                # Keep backward compatibility: val/test both use val fold.
                self._index = train_index if partition == "train" else val_index
            else:
                raise ValueError(
                    f"split_strategy must be one of {{'three_way', 'kfold'}}, got {split_strategy}"
                )

        self.instance_transforms = instance_transforms
        self.use_mmap = bool(use_mmap)
        self.cache_in_memory = bool(cache_in_memory)
        self._memory_cache = None
        if self.cache_in_memory:
            # Determine appropriate number of workers
            # Leave at least 1 core free, cap at a reasonable number to avoid memory spikes
            num_workers = max(1, min(mp.cpu_count() - 1, 32))
            
            self._temp_for_mp = self._index
            
            # Use file_system strategy to avoid "received 0 items of ancdata" (file descriptor limit)
            mp.set_sharing_strategy('file_system')
            with mp.Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._load_raw_for_mp, range(len(self._index))),
                        total=len(self._index),
                        desc=f"Caching {partition} cached vectors in memory (MP, workers={num_workers})",
                    )
                )
            self._memory_cache = results
            
    def _load_raw_for_mp(self, ind):
        return self._load_raw(ind)

    def __len__(self):
        return len(self._memory_cache) if self._memory_cache is not None else len(self._index)

    def _load_payload(self, vector_path):
        """
        Load cached payload with backward compatibility for torch versions.
        """
        try:
            if self.use_mmap:
                return torch.load(
                    vector_path,
                    map_location="cpu",
                    mmap=True,
                    weights_only=False,
                )
            return torch.load(
                vector_path,
                map_location="cpu",
                weights_only=False,
            )
        except RuntimeError as error:
            # Fallback for old/non-zip torch.save payloads or mmap-incompatible files.
            if self.use_mmap and "mmap" in str(error).lower():
                try:
                    return torch.load(
                        vector_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                except TypeError:
                    return torch.load(vector_path, map_location="cpu")
            raise
        except TypeError:
            try:
                return torch.load(vector_path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(vector_path, map_location="cpu")

    @staticmethod
    def _to_three_channel_label(label):
        """
        Convert scalar BraTS label map to [TC, WT, ET] channels.
        Accepts [D,H,W] or [1,D,H,W].
        """
        if label.ndim == 4 and label.shape[0] == 1:
            label = label.squeeze(0)
        if label.ndim != 3:
            raise ValueError(f"Unsupported scalar label shape: {tuple(label.shape)}")

        label = label.long()
        tc = (label == 1) | (label == 3)
        wt = (label == 1) | (label == 2) | (label == 3)
        et = label == 3
        return torch.stack([tc, wt, et], dim=0).float()

    def __getitem__(self, ind):
        if self._memory_cache is not None:
            cached = self._memory_cache[ind]
            sample = {
                "image": cached["image"].clone(),
                "label": cached["label"].clone(),
                "case_id": cached["case_id"],
            }
        else:
            sample = self._load_raw(ind)
            
        return self._apply_transforms_and_normalize(sample)

    def _load_raw(self, ind):
        record = self._index[ind]
        vector_path = Path(record["vector_path"]).expanduser().resolve()
        sample = self._load_payload(vector_path)

        image = torch.as_tensor(sample["image"])
        label = torch.as_tensor(sample["label"])

        if label.ndim == 3:
            label = label.unsqueeze(0)
        elif label.ndim == 4 and label.shape[0] in {1, 3}:
            pass
        else:
            raise ValueError(
                f"Unsupported cached label shape {tuple(label.shape)} in {vector_path}"
            )

        return {
            "image": image,
            "label": label,
            "case_id": record.get("case_id", vector_path.stem),
        }

    def _apply_transforms_and_normalize(self, sample):
        if self.instance_transforms is not None:
            sample = self.instance_transforms(sample)

        sample["image"] = torch.as_tensor(sample["image"]).float()
        final_label = torch.as_tensor(sample["label"])
        
        if final_label.ndim == 3 or (final_label.ndim == 4 and final_label.shape[0] == 1):
            final_label = self._to_three_channel_label(final_label)
        elif final_label.ndim == 4 and final_label.shape[0] == 3:
            final_label = final_label.float()
        else:
            raise ValueError(
                f"Unsupported transformed label shape {tuple(final_label.shape)}"
            )
            
        sample["label"] = final_label
        return sample
