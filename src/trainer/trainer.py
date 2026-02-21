import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def _model_for_inference(self):
        """
        Use plain module for eval to avoid DDP forward-time collectives.
        This is important for sliding-window inference where different
        samples can trigger different numbers of window forwards per rank.
        """
        if (not self.is_train) and hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _forward_model(self, batch):
        """
        Forward model and return output dict.

        Training:
        - uses direct full-volume forward
        - keeps auxiliary outputs (e.g., deep supervision logits)

        Validation/testing:
        - optionally uses MONAI sliding-window inference
        - returns logits only
        """
        image = batch["image"]
        model = self._model_for_inference()
        use_sw = False
        if not self.is_train:
            eval_mode = str(getattr(self, "current_eval_mode", "full")).lower()
            if eval_mode == "quick":
                quick_cfg = getattr(self, "current_eval_quick_cfg", {}) or {}
                use_sw = bool(quick_cfg.get("use_sliding_window", False))
            else:
                use_sw = bool(self.cfg_trainer.get("use_sliding_window_inference", False))
        if not use_sw:
            outputs = model(**batch)
            if not isinstance(outputs, dict) or "logits" not in outputs:
                raise ValueError(
                    "Model forward must return a dict containing key 'logits'."
                )
            return outputs

        roi_size = tuple(self.cfg_trainer.get("sw_roi_size", [96, 96, 96]))
        sw_batch_size = int(self.cfg_trainer.get("sw_batch_size", 1))
        overlap = float(self.cfg_trainer.get("sw_overlap", 0.5))

        logits = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=lambda x: model(image=x)["logits"],
            overlap=overlap,
        )
        return {"logits": logits}

    @staticmethod
    def _pad_to_roi_size(tensor, roi_size):
        """
        Pad tensor [B, C, D, H, W] to at least roi_size on spatial dims.
        Padding is appended to the end of each spatial axis.
        """
        if tensor.ndim != 5:
            raise ValueError(f"Expected 5D tensor, got shape={tuple(tensor.shape)}")
        d, h, w = tensor.shape[2:]
        pad_d = max(0, int(roi_size[0]) - int(d))
        pad_h = max(0, int(roi_size[1]) - int(h))
        pad_w = max(0, int(roi_size[2]) - int(w))
        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return tensor
        return F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d))

    def _prepare_quick_validation_batch(self, batch):
        """
        Build random (or center) ROI patches for quick validation.
        """
        quick_cfg = getattr(self, "current_eval_quick_cfg", {}) or {}
        roi_size = quick_cfg.get("roi_size", self.cfg_trainer.get("sw_roi_size", [96, 96, 96]))
        roi_size = tuple(int(x) for x in roi_size)
        if len(roi_size) != 3:
            raise ValueError(
                "quick validation roi_size must have 3 ints, "
                f"got {roi_size!r}"
            )
        random_crop = bool(quick_cfg.get("random_crop", True))
        pad_if_needed = bool(quick_cfg.get("pad_if_needed", True))

        image = batch["image"]
        label = batch["label"]
        if image.ndim != 5 or label.ndim != 5:
            return batch
        if image.shape[0] != label.shape[0]:
            raise ValueError(
                "image and label batch size mismatch in quick validation: "
                f"{image.shape[0]} vs {label.shape[0]}"
            )

        cropped_images = []
        cropped_labels = []
        for idx in range(image.shape[0]):
            img = image[idx : idx + 1]
            lbl = label[idx : idx + 1]

            slices = []
            for dim_size, target in zip(img.shape[2:], roi_size):
                if int(dim_size) > int(target):
                    if random_crop:
                        start = int(
                            torch.randint(
                                low=0,
                                high=int(dim_size) - int(target) + 1,
                                size=(1,),
                                device=img.device,
                            ).item()
                        )
                    else:
                        start = (int(dim_size) - int(target)) // 2
                    end = start + int(target)
                else:
                    start = 0
                    end = int(dim_size)
                slices.append(slice(start, end))

            img_crop = img[:, :, slices[0], slices[1], slices[2]]
            lbl_crop = lbl[:, :, slices[0], slices[1], slices[2]]

            if pad_if_needed:
                img_crop = self._pad_to_roi_size(img_crop, roi_size)
                lbl_crop = self._pad_to_roi_size(lbl_crop, roi_size)

            cropped_images.append(img_crop)
            cropped_labels.append(lbl_crop)

        batch["image"] = torch.cat(cropped_images, dim=0)
        batch["label"] = torch.cat(cropped_labels, dim=0)
        return batch

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
        if (not self.is_train) and str(getattr(self, "current_eval_mode", "full")).lower() == "quick":
            batch = self._prepare_quick_validation_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with torch.autocast(
            device_type=self.autocast_device_type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            model_outputs = self._forward_model(batch)
            batch.update(model_outputs)
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
