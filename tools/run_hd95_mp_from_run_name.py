#!/usr/bin/env python3
"""
Multi-process HD95 + Dice evaluation from saved/<run_name>.

Design:
- Auto-load config from saved/<run_name>/config.yaml
- Auto-select checkpoint in run dir:
  1) best_model.pth
  2) model_best.pth
  3) latest checkpoint-epoch*.pth
- Build target partition dataset (default: test), split by case_id into N shards
- Spawn N worker processes; each process loads model + ckpt, runs inference on its shard
- Main process aggregates MetricTracker totals/counts from all workers

Notes:
- Metric semantics match existing trainer script style (batch-wise average via MetricTracker).
- For CUDA, each worker loads a full model copy. Ensure enough VRAM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from monai.inferers import sliding_window_inference
from omegaconf import OmegaConf
from scipy.ndimage import label as cc_label
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.collate import collate_fn
from src.datasets.data_utils import get_dataloaders
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from src.utils.monai_compat import patch_monai_numpy_dtype_compat


def _set_track_meta_false() -> None:
    try:
        from monai.data.meta_obj import set_track_meta
    except Exception:
        try:
            from monai.data.meta_tensor import set_track_meta
        except Exception:
            return
    set_track_meta(False)


def _checkpoint_epoch_key(path: Path) -> int:
    match = re.search(r"checkpoint-epoch(\d+)\.pth$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _find_best_checkpoint(run_dir: Path) -> Path:
    for name in ("best_model.pth", "model_best.pth"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate

    candidates = sorted(run_dir.glob("checkpoint-epoch*.pth"), key=_checkpoint_epoch_key)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in '{run_dir}'. "
            "Expected best_model.pth / model_best.pth / checkpoint-epoch*.pth."
        )
    return candidates[-1]


def _load_checkpoint_compat(checkpoint_path: Path, device: str):
    try:
        return torch.load(
            str(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(checkpoint_path), map_location=device)


def _resolve_device(config, cli_device: str) -> str:
    if cli_device != "auto":
        return cli_device
    configured = str(config.trainer.get("device", "auto")).lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured


def _ensure_test_dataset_cfg(config) -> None:
    if "datasets" not in config:
        raise ValueError("config missing 'datasets' section")

    if "test" not in config.datasets:
        if "val" not in config.datasets:
            raise ValueError("config.datasets has neither 'test' nor 'val'")
        config.datasets.test = OmegaConf.create(
            OmegaConf.to_container(config.datasets.val, resolve=False)
        )

    test_cfg = config.datasets.test
    test_cfg.partition = "test"
    test_cfg.split_strategy = "three_way"
    if "val_ratio" not in test_cfg:
        test_cfg.val_ratio = 0.1
    if "test_ratio" not in test_cfg:
        test_cfg.test_ratio = 0.1

    if "transforms" in config and "instance_transforms" in config.transforms:
        test_cfg.instance_transforms = config.transforms.instance_transforms.inference


_STRUCT26 = np.ones((3, 3, 3), dtype=np.uint8)


def _count_cc_and_fragment_ratio(mask: np.ndarray) -> tuple[int, float]:
    labeled, n_cc = cc_label(mask.astype(np.uint8), structure=_STRUCT26)
    if n_cc <= 0:
        return 0, 0.0
    sizes = np.bincount(labeled.ravel())[1:]
    if sizes.size == 0:
        return int(n_cc), 0.0
    total = int(sizes.sum())
    largest = int(sizes.max())
    frag_ratio = float((total - largest) / max(total, 1))
    return int(n_cc), frag_ratio


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _transform_batch(batch: dict, batch_transforms: dict) -> dict:
    transforms = batch_transforms.get("inference") if batch_transforms is not None else None
    if transforms is None:
        return batch
    for transform_name in transforms.keys():
        transform = transforms[transform_name]
        if transform_name in batch:
            batch[transform_name] = transform(batch[transform_name])
        else:
            transformed = transform(batch)
            if not isinstance(transformed, dict):
                raise ValueError(
                    "Batch-level transform must return dict, got "
                    f"{type(transformed)!r} for key={transform_name!r}"
                )
            batch = transformed
    return batch


def _load_state_dict_to_model(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    checkpoint = _load_checkpoint_compat(checkpoint_path=checkpoint_path, device=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)


def _build_case_ids(dataset) -> List[str]:
    if hasattr(dataset, "_index"):
        index = getattr(dataset, "_index")
        if isinstance(index, list) and index and isinstance(index[0], dict):
            ids = []
            for i, record in enumerate(index):
                cid = record.get("case_id")
                ids.append(str(cid) if cid else f"idx_{i:06d}")
            return ids

    # Fallback: iterate dataset entries.
    ids = []
    for i in range(len(dataset)):
        sample = dataset[i]
        cid = sample.get("case_id") if isinstance(sample, dict) else None
        ids.append(str(cid) if cid else f"idx_{i:06d}")
    return ids


def _build_case_to_indices(dataset) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    if hasattr(dataset, "_index"):
        index = getattr(dataset, "_index")
        if isinstance(index, list) and index and isinstance(index[0], dict):
            for i, record in enumerate(index):
                cid = record.get("case_id")
                key = str(cid) if cid else f"idx_{i:06d}"
                mapping.setdefault(key, []).append(i)
            return mapping

    for i in range(len(dataset)):
        sample = dataset[i]
        cid = sample.get("case_id") if isinstance(sample, dict) else None
        key = str(cid) if cid else f"idx_{i:06d}"
        mapping.setdefault(key, []).append(i)
    return mapping


def _split_case_ids(case_ids: Sequence[str], n_shards: int) -> List[List[str]]:
    shards: List[List[str]] = [[] for _ in range(n_shards)]
    for i, cid in enumerate(case_ids):
        shards[i % n_shards].append(cid)
    return shards


@dataclass
class WorkerResult:
    rank: int
    n_cases: int
    n_batches: int
    totals: Dict[str, float]
    counts: Dict[str, float]
    cc_counts: Dict[str, List[float]]
    frag_ratios: Dict[str, List[float]]


def _worker_main(
    rank: int,
    case_ids: List[str],
    cfg_dict: dict,
    checkpoint_path: str,
    partition: str,
    device: str,
    use_sliding_window: bool,
    sw_roi_size: Sequence[int],
    sw_batch_size: int,
    sw_overlap: float,
    eval_batch_size: int,
    loader_workers_per_proc: int,
    pred_threshold: float,
    output_queue,
) -> None:
    try:
        patch_monai_numpy_dtype_compat()
        _set_track_meta_false()

        # Avoid CPU over-subscription in subprocesses.
        if loader_workers_per_proc <= 0:
            torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))

        config = OmegaConf.create(cfg_dict)

        # Keep worker dataloaders lightweight.
        OmegaConf.set_struct(config, False)
        config.dataloader.num_workers = int(loader_workers_per_proc)
        if int(loader_workers_per_proc) <= 0:
            config.dataloader.persistent_workers = False
            if "prefetch_factor" in config.dataloader:
                config.dataloader.pop("prefetch_factor")
            if "multiprocessing_context" in config.dataloader:
                config.dataloader.pop("multiprocessing_context")
        OmegaConf.set_struct(config, True)

        dataloaders, batch_transforms = get_dataloaders(
            config=config,
            device=device,
            distributed=False,
            rank=0,
            world_size=1,
            distributed_eval=False,
        )
        if partition not in dataloaders:
            raise ValueError(f"partition={partition!r} not found, available={list(dataloaders.keys())}")

        base_dataset = dataloaders[partition].dataset
        case_to_indices = _build_case_to_indices(base_dataset)
        subset_indices: List[int] = []
        for cid in case_ids:
            subset_indices.extend(case_to_indices.get(cid, []))
        subset = Subset(base_dataset, subset_indices)

        loader = DataLoader(
            dataset=subset,
            batch_size=int(eval_batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(loader_workers_per_proc),
            collate_fn=collate_fn,
            pin_memory=(str(device).startswith("cuda")),
            persistent_workers=bool(loader_workers_per_proc > 0),
        )

        model = instantiate(config.model).to(device)
        _load_state_dict_to_model(model=model, checkpoint_path=Path(checkpoint_path), device=device)
        model.eval()

        metrics = instantiate(config.metrics)
        inference_metrics = metrics["inference"]
        tracker = MetricTracker(*[m.name for m in inference_metrics], writer=None)
        cc_counts: Dict[str, List[float]] = {"TC": [], "WT": [], "ET": []}
        frag_ratios: Dict[str, List[float]] = {"TC": [], "WT": [], "ET": []}

        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                n_batches += 1
                batch["image"] = batch["image"].to(device)
                batch["label"] = batch["label"].to(device)

                batch = _transform_batch(batch=batch, batch_transforms=batch_transforms)

                if use_sliding_window:
                    logits = sliding_window_inference(
                        inputs=batch["image"],
                        roi_size=tuple(int(x) for x in sw_roi_size),
                        sw_batch_size=int(sw_batch_size),
                        predictor=lambda x: model(image=x),
                        overlap=float(sw_overlap),
                    )
                    outputs = logits if isinstance(logits, dict) else {"logits": logits}
                else:
                    outputs = model(**batch)

                if not isinstance(outputs, dict) or "logits" not in outputs:
                    raise ValueError("Model output must be dict with key 'logits'.")

                batch.update(outputs)
                probs = torch.sigmoid(batch["logits"])
                pred = (probs > float(pred_threshold)).detach().cpu().numpy().astype(np.uint8)
                num_channels = pred.shape[1]
                for b in range(pred.shape[0]):
                    if num_channels > 0:
                        n_cc, frag = _count_cc_and_fragment_ratio(pred[b, 0].astype(bool))
                        cc_counts["TC"].append(float(n_cc))
                        frag_ratios["TC"].append(float(frag))
                    if num_channels > 1:
                        n_cc, frag = _count_cc_and_fragment_ratio(pred[b, 1].astype(bool))
                        cc_counts["WT"].append(float(n_cc))
                        frag_ratios["WT"].append(float(frag))
                    if num_channels > 2:
                        n_cc, frag = _count_cc_and_fragment_ratio(pred[b, 2].astype(bool))
                        cc_counts["ET"].append(float(n_cc))
                        frag_ratios["ET"].append(float(frag))
                for met in inference_metrics:
                    tracker.update(met.name, met(**batch))

        totals = {k: float(v["total"]) for k, v in tracker._data.items()}
        counts = {k: float(v["counts"]) for k, v in tracker._data.items()}
        output_queue.put(
            WorkerResult(
                rank=int(rank),
                n_cases=len(case_ids),
                n_batches=int(n_batches),
                totals=totals,
                counts=counts,
                cc_counts=cc_counts,
                frag_ratios=frag_ratios,
            )
        )
    except Exception as error:  # pragma: no cover - pass full error to main proc
        output_queue.put({"rank": int(rank), "error": repr(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-process HD95 evaluation from saved/<run_name>.")
    parser.add_argument("--run-name", required=True, help="Run directory name under saved/.")
    parser.add_argument("--save-root", default="saved", help="Root dir containing run folders (default: saved).")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument("--partition", default="test", choices=["val", "test"], help="Evaluation partition.")
    parser.add_argument(
        "--metrics-config",
        default="src/configs/metrics/brats23_seg_hd95.yaml",
        help="Metrics config path. Defaults to HD95+Dice config.",
    )
    parser.add_argument("--usage-ratio", type=float, default=1.0, help="Override datasets.*.usage_ratio in (0,1].")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0...")
    parser.add_argument("--nprocs", type=int, default=2, help="Number of worker processes.")
    parser.add_argument(
        "--devices",
        default=None,
        help="Optional comma-separated device list for workers, e.g. 'cuda:0,cuda:1'.",
    )
    parser.add_argument(
        "--loader-workers-per-proc",
        type=int,
        default=0,
        help="DataLoader workers inside each eval process (default: 0).",
    )
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary prediction when computing CC/fragment stats (default: 0.5).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override eval batch size. Default follows config.dataloader.eval_batch_size or batch_size.",
    )
    parser.add_argument(
        "--start-method",
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method.",
    )
    parser.add_argument("--no-sliding-window", action="store_true", help="Disable sliding-window inference.")
    parser.add_argument("--output-json", default=None, help="Optional output json path.")
    args = parser.parse_args()

    if not (0.0 < float(args.usage_ratio) <= 1.0):
        raise ValueError(f"--usage-ratio must be in (0, 1], got {args.usage_ratio}")
    if int(args.nprocs) < 1:
        raise ValueError(f"--nprocs must be >= 1, got {args.nprocs}")
    if not (0.0 < float(args.pred_threshold) < 1.0):
        raise ValueError(f"--pred-threshold must be in (0, 1), got {args.pred_threshold}")

    run_dir = ROOT_PATH / args.save_root / args.run_name
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if args.checkpoint is None:
        checkpoint_path = _find_best_checkpoint(run_dir)
    else:
        raw_ckpt = Path(args.checkpoint).expanduser()
        checkpoint_path = raw_ckpt if raw_ckpt.is_absolute() else run_dir / raw_ckpt
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    metrics_cfg_path = Path(args.metrics_config)
    if not metrics_cfg_path.is_absolute():
        metrics_cfg_path = ROOT_PATH / metrics_cfg_path
    if not metrics_cfg_path.exists():
        raise FileNotFoundError(f"Metrics config not found: {metrics_cfg_path}")

    patch_monai_numpy_dtype_compat()
    _set_track_meta_false()

    config = OmegaConf.load(config_path)
    metrics_cfg = OmegaConf.load(metrics_cfg_path)

    OmegaConf.set_struct(config, False)
    config.metrics = metrics_cfg
    if args.partition == "test":
        _ensure_test_dataset_cfg(config)

    for split_name in ("train", "val", "test"):
        split_cfg = config.get("datasets", {}).get(split_name)
        if split_cfg is not None:
            split_cfg.usage_ratio = float(args.usage_ratio)

    config.trainer.ddp = {"enabled": False, "distributed_eval": False}
    config.trainer.auto_resume = False
    config.trainer.resume_from = None
    config.trainer.override = False
    config.writer.mode = "offline"
    OmegaConf.set_struct(config, True)

    base_device = _resolve_device(config=config, cli_device=args.device)
    eval_batch_size = int(
        args.eval_batch_size
        if args.eval_batch_size is not None
        else int(config.dataloader.get("eval_batch_size", config.dataloader.batch_size))
    )
    use_sliding_window = not bool(args.no_sliding_window) and bool(
        config.trainer.get("use_sliding_window_inference", True)
    )
    sw_roi_size = tuple(config.trainer.get("sw_roi_size", [96, 96, 96]))
    sw_batch_size = int(config.trainer.get("sw_batch_size", 1))
    sw_overlap = float(config.trainer.get("sw_overlap", 0.5))

    # Build partition once (single-worker) to get deterministic case_id split.
    split_cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    OmegaConf.set_struct(split_cfg, False)
    split_cfg.dataloader.num_workers = 0
    split_cfg.dataloader.persistent_workers = False
    if "prefetch_factor" in split_cfg.dataloader:
        split_cfg.dataloader.pop("prefetch_factor")
    if "multiprocessing_context" in split_cfg.dataloader:
        split_cfg.dataloader.pop("multiprocessing_context")
    OmegaConf.set_struct(split_cfg, True)

    dataloaders, _ = get_dataloaders(
        config=split_cfg,
        device=base_device,
        distributed=False,
        rank=0,
        world_size=1,
        distributed_eval=False,
    )
    if args.partition not in dataloaders:
        raise ValueError(
            f"'{args.partition}' dataloader not found. available={list(dataloaders.keys())}"
        )

    dataset = dataloaders[args.partition].dataset
    case_ids = _build_case_ids(dataset)
    if len(case_ids) == 0:
        raise ValueError(f"No samples in partition={args.partition}.")

    nprocs = min(int(args.nprocs), len(case_ids))
    shards = _split_case_ids(case_ids, nprocs)

    if args.devices:
        worker_devices = [x.strip() for x in str(args.devices).split(",") if x.strip()]
        if not worker_devices:
            worker_devices = [base_device]
    else:
        worker_devices = [base_device]
    if len(worker_devices) < nprocs:
        worker_devices = [worker_devices[i % len(worker_devices)] for i in range(nprocs)]

    logger = logging.getLogger("hd95_mp_eval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    logger.info("run_name=%s", args.run_name)
    logger.info("partition=%s", args.partition)
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("metrics=%s", metrics_cfg_path)
    logger.info("base_device=%s", base_device)
    logger.info("workers=%d", nprocs)
    logger.info("eval_batch_size=%d", eval_batch_size)
    logger.info("sliding_window=%s roi=%s sw_batch_size=%d overlap=%.3f", use_sliding_window, sw_roi_size, sw_batch_size, sw_overlap)

    import multiprocessing as mp

    ctx = mp.get_context(args.start_method)
    queue = ctx.Queue()
    procs = []

    cfg_dict = OmegaConf.to_container(config, resolve=True)
    for rank in range(nprocs):
        proc = ctx.Process(
            target=_worker_main,
            args=(
                rank,
                shards[rank],
                cfg_dict,
                str(checkpoint_path),
                str(args.partition),
                str(worker_devices[rank]),
                bool(use_sliding_window),
                tuple(sw_roi_size),
                int(sw_batch_size),
                float(sw_overlap),
                int(eval_batch_size),
                int(args.loader_workers_per_proc),
                float(args.pred_threshold),
                queue,
            ),
        )
        proc.start()
        procs.append(proc)

    results: List[WorkerResult] = []
    errors = []
    for _ in range(nprocs):
        msg = queue.get()
        if isinstance(msg, dict) and "error" in msg:
            errors.append(msg)
        elif isinstance(msg, WorkerResult):
            results.append(msg)
        else:
            errors.append({"rank": -1, "error": f"Unknown worker message: {msg!r}"})

    for proc in procs:
        proc.join()

    if errors:
        raise RuntimeError(f"One or more workers failed: {errors}")

    if not results:
        raise RuntimeError("No worker results collected.")

    metric_keys = sorted(results[0].totals.keys())
    agg_totals = {k: 0.0 for k in metric_keys}
    agg_counts = {k: 0.0 for k in metric_keys}

    total_cases = 0
    total_batches = 0
    agg_cc_counts: Dict[str, List[float]] = {"TC": [], "WT": [], "ET": []}
    agg_frag_ratios: Dict[str, List[float]] = {"TC": [], "WT": [], "ET": []}
    for item in results:
        total_cases += int(item.n_cases)
        total_batches += int(item.n_batches)
        for k in metric_keys:
            agg_totals[k] += float(item.totals.get(k, 0.0))
            agg_counts[k] += float(item.counts.get(k, 0.0))
        for region_name in ("TC", "WT", "ET"):
            agg_cc_counts[region_name].extend(item.cc_counts.get(region_name, []))
            agg_frag_ratios[region_name].extend(item.frag_ratios.get(region_name, []))

    logs = {
        k: (agg_totals[k] / agg_counts[k] if agg_counts[k] > 0.0 else 0.0)
        for k in metric_keys
    }
    for region_name in ("TC", "WT", "ET"):
        cc_stats = _summary_stats(agg_cc_counts[region_name])
        frag_stats = _summary_stats(agg_frag_ratios[region_name])
        for key, value in cc_stats.items():
            logs[f"CCCount_{region_name}_{key}"] = float(value)
        for key, value in frag_stats.items():
            logs[f"FragRatio_{region_name}_{key}"] = float(value)
    logs["CCFrag_case_count"] = float(
        max(
            len(agg_cc_counts["TC"]),
            len(agg_cc_counts["WT"]),
            len(agg_cc_counts["ET"]),
        )
    )

    print("\n=== Multi-Process HD95 Evaluation Results ===")
    print(f"run_name: {args.run_name}")
    print(f"partition: {args.partition}")
    print(f"checkpoint: {checkpoint_path.name}")
    print(f"usage_ratio: {args.usage_ratio}")
    print(f"nprocs: {nprocs}")
    print(f"total_cases: {total_cases}")
    print(f"total_batches: {total_batches}")
    for key in sorted(logs.keys()):
        print(f"{key}: {logs[key]}")

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = ROOT_PATH / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": args.run_name,
            "partition": args.partition,
            "checkpoint": str(checkpoint_path),
            "usage_ratio": float(args.usage_ratio),
            "pred_threshold": float(args.pred_threshold),
            "nprocs": int(nprocs),
            "devices": worker_devices[:nprocs],
            "metrics_config": str(metrics_cfg_path),
            "total_cases": int(total_cases),
            "total_batches": int(total_batches),
            "logs": {k: float(v) for k, v in logs.items()},
            "workers": [
                {
                    "rank": int(r.rank),
                    "n_cases": int(r.n_cases),
                    "n_batches": int(r.n_batches),
                }
                for r in sorted(results, key=lambda x: x.rank)
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved_json: {out_path}")


if __name__ == "__main__":
    main()
