import os

import torch
import torch.distributed as dist


def _parse_env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def barrier():
    if is_dist_available_and_initialized():
        dist.barrier()


def cleanup_distributed():
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def init_distributed_from_config(config):
    """
    Initialize torch.distributed by trainer.ddp config and torchrun env.

    Returns:
        dict with keys:
            enabled, rank, local_rank, world_size, device, backend
    """
    ddp_cfg = config.trainer.get("ddp", {})
    enabled_cfg = ddp_cfg.get("enabled", "auto")
    if isinstance(enabled_cfg, str):
        enabled_cfg = enabled_cfg.lower()

    world_size_env = _parse_env_int("WORLD_SIZE", 1)
    rank_env = _parse_env_int("RANK", 0)
    local_rank_env = _parse_env_int("LOCAL_RANK", 0)

    if enabled_cfg == "auto":
        ddp_enabled = world_size_env > 1
    else:
        ddp_enabled = bool(enabled_cfg)

    if ddp_enabled and world_size_env <= 1:
        # User can still run single-process without failing hard.
        ddp_enabled = False

    if not ddp_enabled:
        if config.trainer.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.trainer.device
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": device,
            "backend": None,
        }

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank_env)
        default_backend = "nccl"
        device = f"cuda:{local_rank_env}"
    else:
        default_backend = "gloo"
        device = "cpu"

    backend = str(ddp_cfg.get("backend", default_backend)).lower()
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return {
        "enabled": True,
        "rank": rank_env,
        "local_rank": local_rank_env,
        "world_size": world_size_env,
        "device": device,
        "backend": backend,
    }

