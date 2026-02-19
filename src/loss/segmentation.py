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
        squared_pred=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    ):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.gamma = float(gamma)
        self.alpha = alpha

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

    def forward(self, logits, label, **batch):
        dice = self.dice_loss(logits, label)
        focal = self._focal_with_logits(logits, label)
        total = self.dice_weight * dice + self.focal_weight * focal
        return {
            "loss": total,
            "loss_dice": dice.detach(),
            "loss_focal": focal.detach(),
        }
