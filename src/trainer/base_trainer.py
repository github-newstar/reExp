import json
from abc import abstractmethod

import torch
import torch.distributed as dist
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
        rank=0,
        world_size=1,
        is_distributed=False,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.is_distributed = bool(is_distributed)
        self.is_main_process = self.rank == 0
        ddp_cfg = self.cfg_trainer.get("ddp", {})
        # Align with mature frameworks: validation can run on all ranks and then
        # aggregate metrics to rank0 for logging/checkpoint decisions.
        self.distributed_eval = self.is_distributed and bool(
            ddp_cfg.get("distributed_eval", True)
        )

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms
        self.lr_scheduler_step_per = str(
            self.cfg_trainer.get("lr_scheduler_step_per", "batch")
        ).lower()
        if self.lr_scheduler_step_per not in {"batch", "epoch"}:
            raise ValueError(
                "trainer.lr_scheduler_step_per must be 'batch' or 'epoch', "
                f"got {self.lr_scheduler_step_per!r}"
            )
        self.eval_interval = int(self.cfg_trainer.get("eval_interval", 1))
        if self.eval_interval < 1:
            raise ValueError(
                "trainer.eval_interval must be >= 1, "
                f"got {self.eval_interval!r}"
            )

        self.dynamic_eval_cfg = self.cfg_trainer.get("dynamic_eval", {})
        if not self.dynamic_eval_cfg:
            self.dynamic_eval_cfg = self.config.get("model", {}).get("dynamic_eval", {})
        self.dynamic_eval_enabled = bool(self.dynamic_eval_cfg.get("enabled", False))
        self.validation_policy_cfg = self.cfg_trainer.get("validation_policy", {})
        self.validation_policy_enabled = bool(
            self.validation_policy_cfg.get("enabled", False)
        )
        self.default_eval_mode = str(
            self.validation_policy_cfg.get("default_mode", "full")
        ).lower()
        if self.default_eval_mode not in {"quick", "full"}:
            raise ValueError(
                "trainer.validation_policy.default_mode must be 'quick' or 'full', "
                f"got {self.default_eval_mode!r}"
            )
        self.current_eval_mode = "full"
        self.current_eval_quick_cfg = {}
        self._eval_history = []
        self._saved_checkpoint_epochs = set()
        self._last_epoch_eval_ran = False
        self._last_epoch_eval_mode = "full"

        # mixed precision setup (AMP)
        self.amp_enabled = bool(self.cfg_trainer.get("amp", False))
        self.amp_dtype_str = str(self.cfg_trainer.get("amp_dtype", "fp16")).lower()
        if self.amp_dtype_str in ("fp16", "float16"):
            self.amp_dtype = torch.float16
        elif self.amp_dtype_str in ("bf16", "bfloat16"):
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Unsupported trainer.amp_dtype={self.amp_dtype_str}. "
                "Use 'fp16' or 'bf16'."
            )

        device_str = str(self.device)
        if device_str.startswith("cuda"):
            self.autocast_device_type = "cuda"
        elif device_str.startswith("mps"):
            self.autocast_device_type = "mps"
        else:
            self.autocast_device_type = "cpu"

        if (
            self.amp_enabled
            and self.autocast_device_type == "cpu"
            and self.amp_dtype != torch.bfloat16
        ):
            self.logger.warning(
                "CPU autocast only supports bf16 reliably. "
                "Disabling AMP because amp_dtype is not bf16."
            )
            self.amp_enabled = False
        self.use_grad_scaler = (
            self.amp_enabled
            and self.autocast_device_type == "cuda"
            and self.amp_dtype == torch.float16
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        # By default, train-time evaluation only runs on validation set.
        # test should be executed separately via inference script.
        eval_partitions = list(self.cfg_trainer.get("eval_partitions", ["val"]))
        if len(eval_partitions) == 0:
            self.logger.warning(
                "trainer.eval_partitions is empty. No evaluation partition will run during training."
            )
        unknown_parts = [p for p in eval_partitions if p not in dataloaders]
        if unknown_parts:
            raise ValueError(
                f"Unknown trainer.eval_partitions={unknown_parts}. "
                f"Available dataloaders are: {list(dataloaders.keys())}"
            )
        self.evaluation_dataloaders = {k: dataloaders[k] for k in eval_partitions}

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def _build_et_threshold_search_state(self, part):
        """
        Build ET threshold-search state from config.

        The search runs on already-computed logits from the same evaluation pass
        (no repeated model forward), which keeps overhead low.
        """
        cfg = self.cfg_trainer.get("et_threshold_search")
        if cfg is None or not bool(cfg.get("enabled", False)):
            return None

        parts = list(cfg.get("parts", ["val"]))
        if part not in parts:
            return None

        thresholds = [float(x) for x in cfg.get("thresholds", [0.3, 0.4, 0.5, 0.6])]
        if len(thresholds) == 0:
            return None

        return {
            "thresholds": thresholds,
            "channel_index": int(cfg.get("channel_index", 2)),
            "apply_sigmoid": bool(cfg.get("apply_sigmoid", True)),
            "smooth": float(cfg.get("smooth", 1e-5)),
            "log_all": bool(cfg.get("log_all", False)),
            "intersection": [0.0 for _ in thresholds],
            "pred_sum": [0.0 for _ in thresholds],
            "target_sum": 0.0,
        }

    @staticmethod
    def _update_et_threshold_search_state(state, batch):
        logits = batch["logits"].detach().float()
        label = batch["label"].detach().float()

        channel_index = state["channel_index"]
        if logits.ndim < 2 or label.ndim < 2:
            raise ValueError("Expected logits/label with channel dimension for ET search.")
        if channel_index >= logits.shape[1] or channel_index >= label.shape[1]:
            raise ValueError(
                f"ET channel index {channel_index} is out of range for logits={tuple(logits.shape)} "
                f"and label={tuple(label.shape)}"
            )

        if state["apply_sigmoid"]:
            probs = torch.sigmoid(logits[:, channel_index])
        else:
            probs = logits[:, channel_index]
        target = label[:, channel_index]

        target_cpu = target.cpu()
        probs_cpu = probs.cpu()
        state["target_sum"] += float(target_cpu.sum().item())

        for idx, threshold in enumerate(state["thresholds"]):
            pred = (probs_cpu > threshold).float()
            state["intersection"][idx] += float((pred * target_cpu).sum().item())
            state["pred_sum"][idx] += float(pred.sum().item())

    @staticmethod
    def _finalize_et_threshold_search_state(state):
        smooth = state["smooth"]
        dice_scores = []
        for idx in range(len(state["thresholds"])):
            numerator = 2.0 * state["intersection"][idx] + smooth
            denominator = state["pred_sum"][idx] + state["target_sum"] + smooth
            dice_scores.append(float(numerator / denominator))

        best_idx = max(range(len(dice_scores)), key=lambda i: dice_scores[i])
        logs = {
            "ET_Search_BestDice": dice_scores[best_idx],
            "ET_Search_BestThreshold": float(state["thresholds"][best_idx]),
        }

        if state["log_all"]:
            for threshold, score in zip(state["thresholds"], dice_scores):
                key = f"ET_Search_Dice_t{int(round(threshold * 100)):03d}"
                logs[key] = float(score)

        return logs

    def _sync_et_threshold_search_state(self, state):
        """
        Synchronize ET threshold-search accumulators across ranks.
        """
        if state is None:
            return state
        if not self.distributed_eval:
            return state
        if not (self.is_distributed and dist.is_available() and dist.is_initialized()):
            return state

        device = self.device if str(self.device).startswith("cuda") else "cpu"
        intersection = torch.tensor(
            state["intersection"], dtype=torch.float64, device=device
        )
        pred_sum = torch.tensor(state["pred_sum"], dtype=torch.float64, device=device)
        target_sum = torch.tensor([state["target_sum"]], dtype=torch.float64, device=device)

        dist.all_reduce(intersection, op=dist.ReduceOp.SUM)
        dist.all_reduce(pred_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(target_sum, op=dist.ReduceOp.SUM)

        state["intersection"] = intersection.cpu().tolist()
        state["pred_sum"] = pred_sum.cpu().tolist()
        state["target_sum"] = float(target_sum.item())
        return state

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _resolve_legacy_eval_schedule(self, epoch):
        """
        Resolve evaluation schedule when validation_policy is disabled.
        """
        is_eval_epoch = False
        current_interval = self.eval_interval
        mode = "full"

        if epoch == 1:
            is_eval_epoch = True
        elif self.dynamic_eval_enabled:
            progress = epoch / self.epochs

            # For short runs (<= 100 epochs)
            if self.epochs <= 100:
                if progress <= 0.5:
                    current_interval = 5
                elif progress <= 0.8:
                    current_interval = 2
                else:
                    current_interval = 1
            # For long runs (> 100 epochs)
            else:
                if progress <= 0.5:
                    current_interval = 20
                elif progress <= 0.8:
                    current_interval = 10
                elif progress <= 0.9:
                    current_interval = 5
                else:
                    current_interval = 2

            is_eval_epoch = epoch % current_interval == 0
        else:
            is_eval_epoch = epoch % self.eval_interval == 0

        return {
            "run_eval": bool(is_eval_epoch),
            "mode": mode,
            "interval": int(current_interval),
            "reason": "legacy_dynamic_eval" if self.dynamic_eval_enabled else "eval_interval",
        }

    def _resolve_policy_eval_schedule(self, epoch):
        """
        Resolve evaluation schedule/mode from trainer.validation_policy.
        """
        cfg = self.validation_policy_cfg
        run_eval = False
        mode = self.default_eval_mode

        always_eval_epoch1 = bool(cfg.get("always_eval_epoch1", True))
        if always_eval_epoch1 and epoch == 1:
            run_eval = True

        progress = epoch / self.epochs if self.epochs > 0 else 1.0
        phases = list(cfg.get("phases", []))
        selected_phase = None
        for phase in phases:
            max_ratio = float(phase.get("max_epoch_ratio", 1.0))
            if progress <= max_ratio:
                selected_phase = phase
                break
        if selected_phase is None and phases:
            selected_phase = phases[-1]

        if selected_phase is not None:
            mode = str(selected_phase.get("mode", mode)).lower()
            interval = int(selected_phase.get("interval", self.eval_interval))
        else:
            mode = str(cfg.get("mode", mode)).lower()
            interval = int(cfg.get("interval", self.eval_interval))

        if mode not in {"quick", "full"}:
            raise ValueError(
                "trainer.validation_policy mode must be 'quick' or 'full', "
                f"got {mode!r}"
            )

        if interval > 0 and epoch % interval == 0:
            run_eval = True

        full_eval_cfg = cfg.get("full_eval", {})
        if bool(full_eval_cfg.get("enabled", False)):
            full_interval = int(full_eval_cfg.get("interval", 0))
            full_epochs = {int(x) for x in list(full_eval_cfg.get("epochs", []))}
            force_full = (full_interval > 0 and epoch % full_interval == 0) or (
                epoch in full_epochs
            )
            if force_full:
                run_eval = True
                mode = "full"

        return {
            "run_eval": bool(run_eval),
            "mode": mode,
            "interval": int(interval),
            "reason": "validation_policy",
        }

    def _resolve_eval_schedule(self, epoch):
        if self.validation_policy_enabled:
            return self._resolve_policy_eval_schedule(epoch)
        return self._resolve_legacy_eval_schedule(epoch)

    def _resolve_eval_max_batches(self, eval_mode):
        if eval_mode != "quick":
            return None
        quick_cfg = self.validation_policy_cfg.get("quick", {})
        max_batches = quick_cfg.get("max_batches")
        if max_batches is None:
            return None
        max_batches = int(max_batches)
        if max_batches < 1:
            raise ValueError(
                "trainer.validation_policy.quick.max_batches must be >= 1 or null, "
                f"got {max_batches!r}"
            )
        return max_batches

    def _record_eval_history(self, epoch, part, eval_mode, logs):
        self._eval_history.append(
            {
                "epoch": int(epoch),
                "part": str(part),
                "mode": str(eval_mode),
                "metrics": {k: float(v) for k, v in logs.items()},
            }
        )

    def _run_post_training_full_eval(self):
        """
        Optionally run full validation on top-k checkpoints selected by quick-eval scores.
        """
        if not self.validation_policy_enabled:
            return
        cfg = self.validation_policy_cfg.get("post_training_full_eval", {})
        if not bool(cfg.get("enabled", False)):
            return
        if not self.is_main_process:
            return
        if self.is_distributed:
            self.logger.warning(
                "post_training_full_eval is skipped in distributed mode. "
                "Run single-process validation for top-k checkpoints if needed."
            )
            return

        metric_key = str(cfg.get("metric", "val_MeanDice"))
        if "_" not in metric_key:
            self.logger.warning(
                "post_training_full_eval.metric should look like 'val_MeanDice'. "
                f"Got {metric_key!r}. Skipping."
            )
            return
        metric_part, metric_name = metric_key.split("_", 1)
        if metric_part not in self.evaluation_dataloaders:
            self.logger.warning(
                f"post_training_full_eval partition {metric_part!r} is not available in "
                f"evaluation dataloaders {list(self.evaluation_dataloaders.keys())}. Skipping."
            )
            return

        source_modes = [str(x).lower() for x in list(cfg.get("source_modes", ["quick"]))]
        ranking_mode = str(cfg.get("mode", "max")).lower()
        if ranking_mode not in {"max", "min"}:
            raise ValueError(
                "trainer.validation_policy.post_training_full_eval.mode must be 'max' or 'min', "
                f"got {ranking_mode!r}"
            )

        top_k = max(1, int(cfg.get("top_k", 5)))
        by_epoch = {}
        for record in self._eval_history:
            if record["part"] != metric_part:
                continue
            if source_modes and record["mode"] not in source_modes:
                continue
            if metric_name not in record["metrics"]:
                continue
            by_epoch[int(record["epoch"])] = float(record["metrics"][metric_name])

        if not by_epoch:
            self.logger.warning(
                "No evaluation history found for post_training_full_eval. Skipping."
            )
            return

        ranked_epochs = sorted(
            by_epoch.items(),
            key=lambda item: item[1],
            reverse=ranking_mode == "max",
        )
        selected = []
        for epoch, score in ranked_epochs:
            checkpoint_path = self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth"
            if checkpoint_path.exists():
                selected.append((epoch, score, checkpoint_path))
            if len(selected) >= top_k:
                break

        if not selected:
            self.logger.warning(
                "No checkpoint files found for selected top-k epochs. "
                "Set save_period smaller or keep candidate checkpoints."
            )
            return

        eval_parts = list(cfg.get("partitions", [metric_part]))
        unknown_parts = [p for p in eval_parts if p not in self.evaluation_dataloaders]
        if unknown_parts:
            raise ValueError(
                "trainer.validation_policy.post_training_full_eval.partitions contains "
                f"unknown partitions: {unknown_parts}"
            )

        model_ref = self._model_for_state_dict()
        summary = {
            "metric": metric_key,
            "ranking_mode": ranking_mode,
            "source_modes": source_modes,
            "top_k": top_k,
            "candidates": [
                {"epoch": int(epoch), "score": float(score), "checkpoint": str(path)}
                for epoch, score, path in selected
            ],
            "full_eval_results": [],
        }

        self.logger.info(
            "Starting post-training full evaluation on top-%d checkpoints.", len(selected)
        )
        for epoch, quick_score, checkpoint_path in selected:
            checkpoint = torch.load(str(checkpoint_path), self.device)
            model_ref.load_state_dict(checkpoint["state_dict"])
            candidate_result = {
                "epoch": int(epoch),
                "quick_score": float(quick_score),
                "checkpoint": str(checkpoint_path),
                "metrics": {},
            }
            for part in eval_parts:
                logs = self._evaluation_epoch(
                    epoch=epoch,
                    part=part,
                    dataloader=self.evaluation_dataloaders[part],
                    eval_mode="full",
                    max_batches=None,
                )
                for name, value in logs.items():
                    candidate_result["metrics"][f"{part}_{name}"] = float(value)
            summary["full_eval_results"].append(candidate_result)

        summary_path = self.checkpoint_dir / str(
            cfg.get("save_summary_path", "post_full_eval_summary.json")
        )
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self.logger.info(
            "Post-training full evaluation finished. Summary saved to %s", summary_path
        )

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            metric_available = self.mnt_mode == "off" or self.mnt_metric in logs
            if self.is_main_process:
                # print logged information to the screen
                for key, value in logs.items():
                    self.logger.info(f"    {key:15s}: {value}")

                if metric_available:
                    # evaluate model performance according to configured metric,
                    # save best checkpoint as model_best
                    best, stop_process, not_improved_count = self._monitor_performance(
                        logs, not_improved_count
                    )
                else:
                    # Skip monitoring on epochs without evaluation to keep
                    # early-stop logic valid when eval runs every N epochs.
                    best = False
                    stop_process = False
                    if self.mnt_mode != "off":
                        self.logger.info(
                            "Skipping monitor this epoch because "
                            f"'{self.mnt_metric}' is unavailable."
                        )
            else:
                best = False
                stop_process = False

            best, stop_process = self._sync_control_flags(best, stop_process)

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)
            force_candidate_save = False
            if self.validation_policy_enabled and self._last_epoch_eval_ran:
                post_cfg = self.validation_policy_cfg.get("post_training_full_eval", {})
                if bool(post_cfg.get("enabled", False)) and bool(
                    post_cfg.get("save_candidates", False)
                ):
                    source_modes = [
                        str(x).lower()
                        for x in list(post_cfg.get("source_modes", ["quick"]))
                    ]
                    if (not source_modes) or (self._last_epoch_eval_mode in source_modes):
                        metric_key = str(post_cfg.get("metric", "val_MeanDice"))
                        if metric_key in logs:
                            force_candidate_save = True
            if force_candidate_save and int(epoch) not in self._saved_checkpoint_epochs:
                self._save_checkpoint(epoch, save_best=False, only_best=False)

            if self.is_distributed and dist.is_available() and dist.is_initialized():
                dist.barrier()

            if stop_process:  # early_stop
                break

        self._run_post_training_full_eval()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        train_sampler = getattr(self.train_dataloader, "sampler", None)
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        last_train_metrics = self.train_metrics.result()
        for batch_idx, batch in enumerate(
            tqdm(
                self.train_dataloader,
                desc="train",
                total=self.epoch_len,
                disable=not self.is_main_process,
            )
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        if self.lr_scheduler is not None and self.lr_scheduler_step_per == "epoch":
            self.lr_scheduler.step()

        # Run validation:
        # - non-distributed: regular single-process eval
        # - distributed_eval=True: all ranks evaluate their shard
        # - distributed_eval=False: only rank0 evaluates
        run_eval_on_this_rank = (
            (not self.is_distributed) or self.distributed_eval or self.is_main_process
        )
        eval_schedule = self._resolve_eval_schedule(epoch)
        run_eval_this_epoch = (
            len(self.evaluation_dataloaders) > 0 and eval_schedule["run_eval"]
        )
        eval_mode = eval_schedule["mode"]
        eval_max_batches = self._resolve_eval_max_batches(eval_mode=eval_mode)

        if run_eval_on_this_rank and run_eval_this_epoch:
            for part, dataloader in self.evaluation_dataloaders.items():
                val_logs = self._evaluation_epoch(
                    epoch=epoch,
                    part=part,
                    dataloader=dataloader,
                    eval_mode=eval_mode,
                    max_batches=eval_max_batches,
                )
                logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})
                self._record_eval_history(
                    epoch=epoch, part=part, eval_mode=eval_mode, logs=val_logs
                )
        elif run_eval_on_this_rank and len(self.evaluation_dataloaders) > 0:
            self.logger.info(
                "Skip evaluation at epoch %d: reason=%s, interval=%s, mode=%s",
                epoch,
                eval_schedule["reason"],
                eval_schedule["interval"],
                eval_schedule["mode"],
            )

        self._last_epoch_eval_ran = bool(run_eval_this_epoch)
        self._last_epoch_eval_mode = str(eval_mode)
        return logs

    def _evaluation_epoch(self, epoch, part, dataloader, eval_mode="full", max_batches=None):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.current_eval_mode = str(eval_mode).lower()
        if self.current_eval_mode not in {"quick", "full"}:
            raise ValueError(
                f"Unknown eval_mode={self.current_eval_mode!r}. Expected 'quick' or 'full'."
            )
        self.current_eval_quick_cfg = (
            dict(self.validation_policy_cfg.get("quick", {}))
            if self.current_eval_mode == "quick"
            else {}
        )
        self.model.eval()
        self.evaluation_metrics.reset()
        eval_sampler = getattr(dataloader, "sampler", None)
        if eval_sampler is not None and hasattr(eval_sampler, "set_epoch"):
            eval_sampler.set_epoch(epoch)
        et_search_state = self._build_et_threshold_search_state(part=part)
        last_batch_idx = None
        last_batch = None
        total_batches = len(dataloader)
        if max_batches is not None:
            total_batches = min(total_batches, int(max_batches))
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=f"{part}:{self.current_eval_mode}",
                total=total_batches,
                disable=not self.is_main_process,
            ):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
                last_batch_idx = batch_idx
                last_batch = batch
                if et_search_state is not None:
                    self._update_et_threshold_search_state(et_search_state, batch)
            if self.distributed_eval:
                self.evaluation_metrics.sync_between_processes(device=self.device)
                et_search_state = self._sync_et_threshold_search_state(et_search_state)
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            if last_batch_idx is not None and last_batch is not None:
                self._log_batch(
                    last_batch_idx, last_batch, part
                )  # log only the last batch during inference
            if last_batch_idx is None:
                self.logger.warning(
                    "No batches were evaluated for part=%s, mode=%s (max_batches=%s).",
                    part,
                    self.current_eval_mode,
                    max_batches,
                )

        logs = self.evaluation_metrics.result()
        if et_search_state is not None:
            et_logs = self._finalize_et_threshold_search_state(et_search_state)
            logs.update(et_logs)
            for key, value in et_logs.items():
                self.writer.add_scalar(key, value)
        self.current_eval_mode = "full"
        self.current_eval_quick_cfg = {}

        return logs

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode == "off":
            return best, stop_process, not_improved_count

        try:
            # check whether model performance improved or not,
            # according to specified metric(mnt_metric)
            if self.mnt_mode == "min":
                improved = logs[self.mnt_metric] <= self.mnt_best
            elif self.mnt_mode == "max":
                improved = logs[self.mnt_metric] >= self.mnt_best
            else:
                improved = False
        except KeyError:
            self.logger.warning(
                f"Warning: Metric '{self.mnt_metric}' is not found. "
                "Model performance monitoring is disabled for current run."
            )
            # When no evaluation is configured (e.g. eval_partitions=[]),
            # disable monitor gracefully and continue training.
            self.mnt_mode = "off"
            return best, stop_process, not_improved_count

        if improved:
            self.mnt_best = logs[self.mnt_metric]
            not_improved_count = 0
            best = True
        else:
            not_improved_count += 1

        if not_improved_count >= self.early_stop:
            self.logger.info(
                "Validation performance didn't improve for {} epochs. "
                "Training stops.".format(self.early_stop)
            )
            stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        if not self.is_main_process:
            return

        model_to_save = self._model_for_state_dict()
        arch = type(model_to_save).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
            if self.lr_scheduler is not None
            else None,
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            self._saved_checkpoint_epochs.add(int(epoch))
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self._model_for_state_dict().load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.lr_scheduler is not None and checkpoint["lr_scheduler"] is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self._model_for_state_dict().load_state_dict(checkpoint["state_dict"])
        else:
            self._model_for_state_dict().load_state_dict(checkpoint)

    def _model_for_state_dict(self):
        """
        Return the underlying model to save/load state_dict.
        Works for both plain nn.Module and DistributedDataParallel.
        """
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _sync_control_flags(self, best, stop_process):
        """
        Synchronize control flags (best/early-stop) from rank 0 to all ranks.
        """
        if not (self.is_distributed and dist.is_available() and dist.is_initialized()):
            return best, stop_process

        device = self.device if str(self.device).startswith("cuda") else "cpu"
        flags = torch.tensor(
            [int(bool(best)), int(bool(stop_process))],
            dtype=torch.int32,
            device=device,
        )
        dist.broadcast(flags, src=0)
        return bool(flags[0].item()), bool(flags[1].item())
