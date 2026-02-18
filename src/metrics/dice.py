import torch

from src.metrics.base_metric import BaseMetric


def compute_dice_per_channel(prediction, target, smooth=1e-5):
    """
    Compute Dice score per channel for batched segmentation maps.

    Args:
        prediction (Tensor): binary prediction in shape [B, C, ...].
        target (Tensor): binary target in shape [B, C, ...].
        smooth (float): smoothing factor to avoid division by zero.
    Returns:
        Tensor: per-channel Dice score, shape [C].
    """
    prediction = prediction.float()
    target = target.float()

    prediction_flat = prediction.flatten(start_dim=2)
    target_flat = target.flatten(start_dim=2)

    intersection = (prediction_flat * target_flat).sum(dim=2)
    prediction_sum = prediction_flat.sum(dim=2)
    target_sum = target_flat.sum(dim=2)

    dice = (2.0 * intersection + smooth) / (prediction_sum + target_sum + smooth)
    return dice.mean(dim=0)


class MeanDiceMetric(BaseMetric):
    """
    Mean Dice over all channels.
    """

    def __init__(self, threshold=0.5, apply_sigmoid=True, smooth=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.apply_sigmoid = apply_sigmoid
        self.smooth = smooth

    def __call__(self, logits, label, **batch):
        probs = torch.sigmoid(logits) if self.apply_sigmoid else logits
        pred = (probs > self.threshold).float()
        dice_per_channel = compute_dice_per_channel(prediction=pred, target=label, smooth=self.smooth)
        return dice_per_channel.mean().item()


class ChannelDiceMetric(BaseMetric):
    """
    Dice score for a selected channel index.
    """

    def __init__(
        self,
        channel_index,
        threshold=0.5,
        apply_sigmoid=True,
        smooth=1e-5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.channel_index = channel_index
        self.threshold = threshold
        self.apply_sigmoid = apply_sigmoid
        self.smooth = smooth

    def __call__(self, logits, label, **batch):
        probs = torch.sigmoid(logits) if self.apply_sigmoid else logits
        pred = (probs > self.threshold).float()
        dice_per_channel = compute_dice_per_channel(prediction=pred, target=label, smooth=self.smooth)
        if not (0 <= self.channel_index < dice_per_channel.shape[0]):
            raise ValueError(
                f"channel_index={self.channel_index} is out of range for {dice_per_channel.shape[0]} channels."
            )
        return dice_per_channel[self.channel_index].item()
