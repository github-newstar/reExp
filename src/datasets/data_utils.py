import logging
from copy import deepcopy
from itertools import repeat

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed

logger = logging.getLogger(__name__)


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(
    config,
    device,
    distributed=False,
    rank=0,
    world_size=1,
    distributed_eval=True,
):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside

    if distributed and world_size < 2:
        raise ValueError(
            f"distributed=True requires world_size >= 2, got world_size={world_size}"
        )

    # dataloaders init
    dataloaders = {}
    train_batch_size = int(config.dataloader.batch_size)
    eval_batch_size = int(config.dataloader.get("eval_batch_size", train_batch_size))
    dataloader_cfg = OmegaConf.to_container(config.dataloader, resolve=True)
    max_ram_gb = dataloader_cfg.pop("max_ram_gb", None)
    max_ram_bytes = None
    if max_ram_gb is not None:
        max_ram_gb = float(max_ram_gb)
        if max_ram_gb <= 0:
            raise ValueError(
                f"dataloader.max_ram_gb must be > 0 when set, got {max_ram_gb}"
            )
        max_ram_bytes = int(max_ram_gb * (1024 ** 3))
    if "eval_batch_size" in dataloader_cfg:
        dataloader_cfg.pop("eval_batch_size")

    def _estimate_object_nbytes(obj):
        # Conservative tensor-only size estimate.
        if hasattr(obj, "numel") and hasattr(obj, "element_size"):
            try:
                return int(obj.numel()) * int(obj.element_size())
            except Exception:
                return 0
        if isinstance(obj, dict):
            return sum(_estimate_object_nbytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_estimate_object_nbytes(v) for v in obj)
        return 0

    def _estimate_inflight_bytes(sample_nbytes, batch_size, num_workers, prefetch_factor, pin_memory):
        # Approximate outstanding batches:
        # - worker queue: num_workers * prefetch_factor
        # - main thread / current batch: +2
        worker_prefetch = max(1, int(num_workers) * int(prefetch_factor))
        outstanding_batches = worker_prefetch + 2
        total = int(sample_nbytes) * int(batch_size) * int(outstanding_batches)
        if bool(pin_memory):
            # Pinned staging can increase resident memory.
            total = int(total * 1.3)
        return total

    def _human_gb(num_bytes):
        return float(num_bytes) / (1024 ** 3)

    def _apply_memory_budget(cfg, dataset, batch_size, partition):
        if max_ram_bytes is None or len(dataset) == 0:
            return cfg

        tuned = deepcopy(cfg)
        num_workers = int(tuned.get("num_workers", 0))
        if num_workers < 0:
            num_workers = 0
            tuned["num_workers"] = 0

        if num_workers == 0:
            tuned.pop("prefetch_factor", None)
            tuned["persistent_workers"] = False
            return tuned

        prefetch_factor = int(tuned.get("prefetch_factor", 2))
        if prefetch_factor < 1:
            prefetch_factor = 1
            tuned["prefetch_factor"] = 1

        pin_memory = bool(tuned.get("pin_memory", False))
        sample = dataset[0]
        sample_nbytes = _estimate_object_nbytes(sample)
        if sample_nbytes <= 0:
            return tuned

        est = _estimate_inflight_bytes(
            sample_nbytes=sample_nbytes,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )

        while est > max_ram_bytes and (prefetch_factor > 1 or num_workers > 0):
            if prefetch_factor > 1:
                prefetch_factor -= 1
                tuned["prefetch_factor"] = prefetch_factor
            else:
                num_workers -= 1
                tuned["num_workers"] = num_workers
                if num_workers == 0:
                    tuned.pop("prefetch_factor", None)
                    tuned["persistent_workers"] = False
                    prefetch_factor = 1

            est = _estimate_inflight_bytes(
                sample_nbytes=sample_nbytes,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
            )

        logger.info(
            "DataLoader memory budget (%s): partition=%s sample≈%.2fMB "
            "budget=%.2fGB estimated=%.2fGB num_workers=%d prefetch_factor=%s",
            max_ram_gb,
            partition,
            sample_nbytes / (1024 ** 2),
            _human_gb(max_ram_bytes),
            _human_gb(est),
            int(tuned.get("num_workers", 0)),
            str(tuned.get("prefetch_factor", "n/a")),
        )
        return tuned

    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]
        is_train_partition = dataset_partition == "train"
        current_batch_size = train_batch_size if is_train_partition else eval_batch_size

        assert current_batch_size <= len(dataset), (
            f"The batch size ({current_batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        sampler = None
        shuffle = is_train_partition

        if distributed:
            if is_train_partition:
                if len(dataset) < world_size:
                    raise ValueError(
                        f"Train dataset too small for DDP: len(dataset)={len(dataset)} < world_size={world_size}"
                    )
                sampler = DistributedSampler(
                    dataset=dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                )
                shuffle = False
            elif distributed_eval:
                # Mature DDP practice: evaluate on all ranks and aggregate metrics.
                # Keep drop_last=False to cover the full validation/test set.
                sampler = DistributedSampler(
                    dataset=dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                shuffle = False

        partition_cfg = _apply_memory_budget(
            cfg=dataloader_cfg,
            dataset=dataset,
            batch_size=current_batch_size,
            partition=dataset_partition,
        )

        partition_dataloader = instantiate(
            partition_cfg,
            dataset=dataset,
            batch_size=current_batch_size,
            collate_fn=collate_fn,
            drop_last=is_train_partition,
            shuffle=shuffle,
            sampler=sampler,
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
