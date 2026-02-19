import torch
from monai.inferers import sliding_window_inference

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def _predict_logits(self, batch):
        """
        Predict logits for a batch.

        Training uses direct full-volume forward.
        Validation/testing can optionally use MONAI sliding-window inference.
        """
        image = batch["image"]
        use_sw = (not self.is_train) and self.cfg_trainer.get(
            "use_sliding_window_inference", False
        )
        if not use_sw:
            return self.model(**batch)["logits"]

        roi_size = tuple(self.cfg_trainer.get("sw_roi_size", [96, 96, 96]))
        sw_batch_size = int(self.cfg_trainer.get("sw_batch_size", 1))
        overlap = float(self.cfg_trainer.get("sw_overlap", 0.5))

        logits = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=lambda x: self.model(image=x)["logits"],
            overlap=overlap,
        )
        return logits

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with torch.autocast(
            device_type=self.autocast_device_type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            batch["logits"] = self._predict_logits(batch)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            if self.use_grad_scaler:
                self.grad_scaler.scale(batch["loss"]).backward()
                self.grad_scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                batch["loss"].backward()  # sum of all losses is always called loss
                self._clip_grad_norm()
                self.optimizer.step()
            if self.lr_scheduler is not None and self.lr_scheduler_step_per == "batch":
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
