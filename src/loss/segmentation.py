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
        self.deep_supervision = bool(deep_supervision)
        self.deep_supervision_weights = tuple(float(w) for w in deep_supervision_weights)

        self.dice_loss = DiceLoss(
            to_onehot_y=False,
            sigmoid=True,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )

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

        return loss.mean()

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
