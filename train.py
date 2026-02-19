import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


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
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

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
    )

    trainer.train()


if __name__ == "__main__":
    main()
