import warnings
import logging
import re
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

try:
    # MONAI compatibility: set_track_meta location differs by version.
    from monai.data.meta_obj import set_track_meta
except Exception:  # pragma: no cover - fallback for older/newer MONAI variants
    try:
        from monai.data.meta_tensor import set_track_meta
    except Exception:
        def set_track_meta(_enabled):  # type: ignore[no-redef]
            return None

from src.datasets.data_utils import get_dataloaders
from src.logger import NullWriter
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.monai_compat import patch_monai_numpy_dtype_compat
from src.utils.io_utils import ROOT_PATH
from src.utils.distributed import (
    barrier,
    cleanup_distributed,
    init_distributed_from_config,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _checkpoint_epoch_key(path: Path) -> int:
    match = re.search(r"checkpoint-epoch(\d+)\.pth$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _find_auto_resume_checkpoint(save_dir: Path) -> tuple[str | None, str | None]:
    """
    Prefer model_best for continuity-by-best; fallback to latest epoch checkpoint.
    """
    best_path = save_dir / "model_best.pth"
    if best_path.exists():
        return best_path.name, "model_best"

    epoch_ckpts = sorted(
        save_dir.glob("checkpoint-epoch*.pth"),
        key=_checkpoint_epoch_key,
    )
    if len(epoch_ckpts) > 0:
        return epoch_ckpts[-1].name, "latest_epoch"
    return None, None


def _maybe_enable_auto_resume(config, is_main_process: bool) -> None:
    if config.trainer.get("resume_from") is not None:
        return
    if not bool(config.trainer.get("auto_resume", True)):
        return

    save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    if not save_dir.exists():
        return

    resume_from, source = _find_auto_resume_checkpoint(save_dir)
    if resume_from is None:
        return

    OmegaConf.set_struct(config, False)
    config.trainer.resume_from = resume_from
    # Avoid deleting existing experiment directory when resuming.
    config.trainer.override = False
    OmegaConf.set_struct(config, True)

    if is_main_process:
        print(
            f"Auto-resume enabled: found {source} checkpoint '{resume_from}' "
            f"under '{save_dir}'."
        )


def _setup_worker_logger(rank):
    logger = logging.getLogger(f"train.rank{rank}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    return logger


def _build_lr_scheduler(config, optimizer, train_dataloader, logger):
    """
    Build LR scheduler with optional warmup.

    Warmup is implemented as:
    LinearLR (warmup) -> main scheduler via SequentialLR.
    """
    main_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    warmup_cfg = config.trainer.get("warmup")
    if warmup_cfg is None or not bool(warmup_cfg.get("enabled", False)):
        return main_scheduler

    step_per = str(config.trainer.get("lr_scheduler_step_per", "batch")).lower()
    if step_per not in {"batch", "epoch"}:
        raise ValueError(
            "trainer.lr_scheduler_step_per must be 'batch' or 'epoch', "
            f"got {step_per!r}"
        )

    n_epochs = int(config.trainer.n_epochs)
    if step_per == "epoch":
        total_steps = n_epochs
        warmup_steps_requested = int(warmup_cfg.get("epochs", 0))
    else:
        epoch_len_cfg = config.trainer.get("epoch_len")
        epoch_len = (
            int(epoch_len_cfg) if epoch_len_cfg is not None else int(len(train_dataloader))
        )
        total_steps = n_epochs * epoch_len
        warmup_steps_requested = int(warmup_cfg.get("epochs", 0)) * epoch_len

    if total_steps <= 1:
        logger.warning("Total scheduler steps <= 1. Warmup is skipped.")
        return main_scheduler

    warmup_steps = min(max(0, warmup_steps_requested), total_steps - 1)
    if warmup_steps == 0:
        return main_scheduler

    start_factor = float(warmup_cfg.get("start_factor", 0.1))
    end_factor = float(warmup_cfg.get("end_factor", 1.0))
    if not (0.0 < start_factor <= 1.0):
        raise ValueError(
            f"trainer.warmup.start_factor must be in (0, 1], got {start_factor}"
        )
    if end_factor <= 0.0:
        raise ValueError(
            f"trainer.warmup.end_factor must be > 0, got {end_factor}"
        )

    main_steps = total_steps - warmup_steps
    if hasattr(main_scheduler, "T_max"):
        main_scheduler.T_max = max(1, int(main_steps))

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )
    logger.info(
        "Warmup enabled: step_per=%s, warmup_steps=%d, main_steps=%d",
        step_per,
        warmup_steps,
        main_steps,
    )
    return scheduler


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    ddp_state = init_distributed_from_config(config)
    is_distributed = bool(ddp_state["enabled"])
    rank = int(ddp_state["rank"])
    world_size = int(ddp_state["world_size"])
    is_main = rank == 0
    device = ddp_state["device"]
    _maybe_enable_auto_resume(config=config, is_main_process=is_main)

    # Apply MONAI dtype compatibility patch early, before transforms are instantiated.
    patch_monai_numpy_dtype_compat()

    # Hard-disable MONAI MetaTensor tracking to avoid known numpy/monai
    # recursion/type-conversion issues in transform meta pipelines.
    set_track_meta(False)
    ddp_cfg = config.trainer.get("ddp", {})
    distributed_eval = bool(ddp_cfg.get("distributed_eval", True))

    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config) if is_main else _setup_worker_logger(rank)
    if is_distributed:
        barrier()

    writer = (
        instantiate(config.writer, logger, project_config)
        if is_main
        else NullWriter(logger, project_config)
    )

    try:
        # setup data_loader instances
        # batch_transforms should be put on device
        dataloaders, batch_transforms = get_dataloaders(
            config,
            device,
            distributed=is_distributed,
            rank=rank,
            world_size=world_size,
            distributed_eval=distributed_eval,
        )

        # build model architecture, then print to console
        model = instantiate(config.model).to(device)
        if is_main:
            logger.info(model)

        if is_distributed:
            find_unused = bool(ddp_cfg.get("find_unused_parameters", False))
            if str(device).startswith("cuda"):
                local_rank = int(ddp_state["local_rank"])
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=find_unused,
                )
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    find_unused_parameters=find_unused,
                )

        # get function handles of loss and metrics
        loss_function = instantiate(config.loss_function).to(device)
        metrics = instantiate(config.metrics)

        # build optimizer, learning rate scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = instantiate(config.optimizer, params=trainable_params)
        lr_scheduler = _build_lr_scheduler(
            config=config,
            optimizer=optimizer,
            train_dataloader=dataloaders["train"],
            logger=logger,
        )

        # epoch_len = number of iterations for iteration-based training
        # epoch_len = None or len(dataloader) for epoch-based training
        epoch_len = config.trainer.get("epoch_len")

        trainer = Trainer(
            model=model,
            criterion=loss_function,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            device=device,
            dataloaders=dataloaders,
            epoch_len=epoch_len,
            logger=logger,
            writer=writer,
            batch_transforms=batch_transforms,
            skip_oom=config.trainer.get("skip_oom", True),
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
