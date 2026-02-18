from monai.losses import DiceLoss
from torch import nn


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
