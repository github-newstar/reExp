from itertools import repeat

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


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


def get_dataloaders(config, device, distributed=False, rank=0, world_size=1):
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
    if "eval_batch_size" in dataloader_cfg:
        dataloader_cfg.pop("eval_batch_size")
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

        if distributed and is_train_partition:
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

        partition_dataloader = instantiate(
            dataloader_cfg,
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
