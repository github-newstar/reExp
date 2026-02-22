import torch
from monai.losses import DiceLoss
from torch import nn
from torch.nn import functional as F


class DiceSegLoss(nn.Module):
    """
    Dice loss wrapper for segmentation tasks.
    """

    def __init__(
        self,
        to_onehot_y=False,
        sigmoid=True,
        squared_pred=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    ):
        super().__init__()
        self.loss_fn = DiceLoss(
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )

    def forward(self, logits, label, **batch):
        return {"loss": self.loss_fn(logits, label)}


class DiceFocalSegLoss(nn.Module):
    """
    Dice + Focal combined loss for highly imbalanced BraTS targets.

    Formula:
    loss = dice_weight * DiceLoss + focal_weight * FocalLoss
    """

    def __init__(
        self,
        dice_weight=1.0,
        focal_weight=1.0,
        gamma=2.0,
        alpha=None,
        channel_weights=None,
        deep_supervision=True,
        deep_supervision_weights=(0.5, 0.25),
        squared_pred=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    ):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.gamma = float(gamma)
        self.alpha = alpha
        self.channel_weights = (
            tuple(float(x) for x in channel_weights) if channel_weights is not None else None
        )
        self.deep_supervision = bool(deep_supervision)
        self.deep_supervision_weights = tuple(float(w) for w in deep_supervision_weights)

        self.dice_loss = DiceLoss(
            to_onehot_y=False,
            sigmoid=True,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            weight=self.channel_weights,
        )

    @staticmethod
    def _weighted_channel_reduce(
        loss_tensor: torch.Tensor,
        channel_weights,
    ) -> torch.Tensor:
        """
        Reduce [B, C, ...] loss tensor to scalar with optional channel weighting.
        """
        if channel_weights is None:
            return loss_tensor.mean()

        if loss_tensor.ndim < 3:
            raise ValueError(
                f"Expected loss tensor with shape [B, C, ...], got {tuple(loss_tensor.shape)}."
            )
        num_channels = int(loss_tensor.shape[1])
        weights = torch.as_tensor(
            channel_weights, dtype=loss_tensor.dtype, device=loss_tensor.device
        )
        if weights.ndim == 0:
            return loss_tensor.mean()
        if int(weights.numel()) != num_channels:
            raise ValueError(
                f"channel_weights expects {num_channels} values, got {int(weights.numel())}."
            )

        reduce_dims = tuple(i for i in range(loss_tensor.ndim) if i != 1)
        per_channel = loss_tensor.mean(dim=reduce_dims)
        return (per_channel * weights).sum() / weights.sum().clamp_min(1e-12)

    def _focal_with_logits(self, logits: torch.Tensor, target: torch.Tensor):
        """
        Binary focal loss over multi-label channels using logits.
        """
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        focal_factor = (1.0 - p_t).pow(self.gamma)

        loss = focal_factor * bce
        if self.alpha is not None:
            alpha = torch.as_tensor(self.alpha, dtype=loss.dtype, device=loss.device)
            if alpha.ndim == 0:
                alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
            else:
                # Per-channel alpha, expected shape [C]
                alpha = alpha.view(1, -1, 1, 1, 1)
                alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
            loss = alpha_t * loss

        return self._weighted_channel_reduce(loss, self.channel_weights)

    def _combined_loss(self, logits: torch.Tensor, label: torch.Tensor):
        dice = self.dice_loss(logits, label)
        focal = self._focal_with_logits(logits, label)
        total = self.dice_weight * dice + self.focal_weight * focal
        return total, dice, focal

    def forward(self, logits, label, aux_logits=None, **batch):
        total, dice, focal = self._combined_loss(logits, label)

        ds_loss = logits.new_tensor(0.0)
        if self.deep_supervision and aux_logits is not None:
            aux_list = [aux_logits] if torch.is_tensor(aux_logits) else list(aux_logits)
            if len(aux_list) > 0:
                for idx, aux in enumerate(aux_list):
                    if idx < len(self.deep_supervision_weights):
                        weight = self.deep_supervision_weights[idx]
                    else:
                        weight = self.deep_supervision_weights[-1] * (
                            0.5 ** (idx - len(self.deep_supervision_weights) + 1)
                        )
                    aux_total, _, _ = self._combined_loss(aux, label)
                    ds_loss = ds_loss + weight * aux_total
                total = total + ds_loss

        return {
            "loss": total,
            "loss_dice": dice.detach(),
            "loss_focal": focal.detach(),
            "loss_ds": ds_loss.detach(),
        }


class GeneralizedDiceFocalSegLoss(nn.Module):
    """
    Generalized Dice Focal Loss (GDFL) inspired by:
    "AutoPET Challenge III: Testing the Robustness of Generalized Dice Focal Loss..."
    (arXiv:2409.10151)

    We implement:
      LGDFL = gdice_weight * LGDL + gfocal_weight * LFL

    with defaults aligned to the paper-style binary setting:
      v0 = 1, v1 = 100, gamma = 2, smooth terms = 1e-5.

    For multi-label BraTS outputs (C channels with sigmoid), each channel is
    treated as a binary segmentation task and then averaged.
    """

    def __init__(
        self,
        gdice_weight=1.0,
        gfocal_weight=1.0,
        gamma=2.0,
        v0=1.0,
        v1=100.0,
        channel_weights=None,
        deep_supervision=True,
        deep_supervision_weights=(0.5, 0.25),
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        # Backward-compatible aliases from DiceFocalSegLoss:
        dice_weight=None,
        focal_weight=None,
        alpha=None,
        squared_pred=None,
        **kwargs,
    ):
        super().__init__()
        # Prefer explicit generalized names; fall back to old names if only they are given.
        if dice_weight is not None and gdice_weight == 1.0:
            gdice_weight = dice_weight
        if focal_weight is not None and gfocal_weight == 1.0:
            gfocal_weight = focal_weight

        self.gdice_weight = float(gdice_weight)
        self.gfocal_weight = float(gfocal_weight)
        self.gamma = float(gamma)
        self.v0 = v0
        self.v1 = v1
        self.channel_weights = (
            tuple(float(x) for x in channel_weights) if channel_weights is not None else None
        )
        self.deep_supervision = bool(deep_supervision)
        self.deep_supervision_weights = tuple(float(w) for w in deep_supervision_weights)
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        # Unused legacy args are intentionally accepted for Hydra merge compatibility.
        _ = alpha, squared_pred, kwargs

    @staticmethod
    def _expand_channel_weight(weight, num_channels, dtype, device):
        weight_t = torch.as_tensor(weight, dtype=dtype, device=device)
        if weight_t.ndim == 0:
            return weight_t
        if weight_t.numel() != int(num_channels):
            raise ValueError(
                f"Channel-wise weight expects {num_channels} values, got {weight_t.numel()}."
            )
        return weight_t.view(1, int(num_channels), 1, 1, 1)

    def _generalized_dice_loss(self, logits: torch.Tensor, target: torch.Tensor):
        """
        Binary generalized Dice per output channel, then averaged.

        For each channel c:
            w_fg = 1 / (sum(g_fg)^2 + eps)
            w_bg = 1 / (sum(g_bg)^2 + eps)
        and then class-weighted Dice over {bg, fg}.
        """
        target = target.float()
        prob = torch.sigmoid(logits)

        prob_flat = prob.flatten(start_dim=2)          # [B, C, V]
        target_flat = target.flatten(start_dim=2)      # [B, C, V]

        prob_bg = 1.0 - prob_flat
        target_bg = 1.0 - target_flat

        fg_volume = target_flat.sum(dim=2)             # [B, C]
        bg_volume = target_bg.sum(dim=2)               # [B, C]

        w_fg = 1.0 / (fg_volume.pow(2) + self.smooth_dr)
        w_bg = 1.0 / (bg_volume.pow(2) + self.smooth_dr)

        num_fg = (prob_flat * target_flat).sum(dim=2)
        num_bg = (prob_bg * target_bg).sum(dim=2)
        den_fg = (prob_flat + target_flat).sum(dim=2)
        den_bg = (prob_bg + target_bg).sum(dim=2)

        numerator = w_fg * num_fg + w_bg * num_bg + self.smooth_nr
        denominator = w_fg * den_fg + w_bg * den_bg + self.smooth_dr

        gdice_per_bc = 1.0 - (numerator / denominator)  # [B, C]
        if self.channel_weights is None:
            return gdice_per_bc.mean()

        num_channels = int(gdice_per_bc.shape[1])
        weights = torch.as_tensor(
            self.channel_weights, dtype=gdice_per_bc.dtype, device=gdice_per_bc.device
        )
        if weights.ndim == 0:
            return gdice_per_bc.mean()
        if int(weights.numel()) != num_channels:
            raise ValueError(
                f"channel_weights expects {num_channels} values, got {int(weights.numel())}."
            )
        per_channel = gdice_per_bc.mean(dim=0)
        return (per_channel * weights).sum() / weights.sum().clamp_min(1e-12)

    def _generalized_focal_with_logits(self, logits: torch.Tensor, target: torch.Tensor):
        """
        Generalized focal term in binary form per channel:
          -v1 * (1-p)^gamma * y * log(p)
          -v0 * p^gamma * (1-y) * log(1-p)
        """
        target = target.float()
        prob = torch.sigmoid(logits)

        num_channels = int(logits.shape[1])
        v0 = self._expand_channel_weight(
            self.v0, num_channels=num_channels, dtype=logits.dtype, device=logits.device
        )
        v1 = self._expand_channel_weight(
            self.v1, num_channels=num_channels, dtype=logits.dtype, device=logits.device
        )

        log_p = F.logsigmoid(logits)
        log_not_p = F.logsigmoid(-logits)

        pos_term = v1 * (1.0 - prob).pow(self.gamma) * target * log_p
        neg_term = v0 * prob.pow(self.gamma) * (1.0 - target) * log_not_p
        focal = -(pos_term + neg_term)
        return DiceFocalSegLoss._weighted_channel_reduce(focal, self.channel_weights)

    def _combined_loss(self, logits: torch.Tensor, label: torch.Tensor):
        gdice = self._generalized_dice_loss(logits, label)
        gfocal = self._generalized_focal_with_logits(logits, label)
        total = self.gdice_weight * gdice + self.gfocal_weight * gfocal
        return total, gdice, gfocal

    def forward(self, logits, label, aux_logits=None, **batch):
        total, gdice, gfocal = self._combined_loss(logits, label)

        ds_loss = logits.new_tensor(0.0)
        if self.deep_supervision and aux_logits is not None:
            aux_list = [aux_logits] if torch.is_tensor(aux_logits) else list(aux_logits)
            if len(aux_list) > 0:
                for idx, aux in enumerate(aux_list):
                    if idx < len(self.deep_supervision_weights):
                        weight = self.deep_supervision_weights[idx]
                    else:
                        weight = self.deep_supervision_weights[-1] * (
                            0.5 ** (idx - len(self.deep_supervision_weights) + 1)
                        )
                    aux_total, _, _ = self._combined_loss(aux, label)
                    ds_loss = ds_loss + weight * aux_total
                total = total + ds_loss

        # Keep old aliases for compatibility with existing writer configs.
        return {
            "loss": total,
            "loss_gdice": gdice.detach(),
            "loss_gfocal": gfocal.detach(),
            "loss_dice": gdice.detach(),
            "loss_focal": gfocal.detach(),
            "loss_ds": ds_loss.detach(),
        }
