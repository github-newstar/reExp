import torch
from monai.metrics import compute_hausdorff_distance

from src.metrics.base_metric import BaseMetric


def _prepare_binary_prediction(logits: torch.Tensor, apply_sigmoid: bool, threshold: float):
    probs = torch.sigmoid(logits) if apply_sigmoid else logits
    return (probs > threshold).float()


def _compute_hd95_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    include_background: bool,
    directed: bool,
):
    """
    Compute HD95 matrix with shape [B, C].
    """
    try:
        hd = compute_hausdorff_distance(
            y_pred=pred,
            y=target.float(),
            include_background=include_background,
            distance_metric="euclidean",
            percentile=95.0,
            directed=directed,
        )
        return hd
    except Exception as error:
        message = str(error)
        if "binary_erosion" in message or "scipy" in message.lower():
            raise RuntimeError(
                "HD95 metric requires scipy. Please install it first "
                "(e.g. `uv sync` after adding scipy, or `uv pip install scipy`)."
            ) from error
        raise


def _finite_mean(values: torch.Tensor, fallback: float):
    finite = torch.isfinite(values)
    if finite.any():
        return values[finite].mean().item()
    return float(fallback)


class MeanHD95Metric(BaseMetric):
    """
    Mean HD95 over all channels and batch samples.

    Note:
    - HD95 is expensive and should only be enabled manually in inference metrics.
    - This implementation ignores non-finite values (common for empty masks).
    """

    def __init__(
        self,
        threshold=0.5,
        apply_sigmoid=True,
        include_background=True,
        directed=False,
        empty_fallback=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.apply_sigmoid = apply_sigmoid
        self.include_background = include_background
        self.directed = directed
        self.empty_fallback = float(empty_fallback)

    def __call__(self, logits, label, **batch):
        pred = _prepare_binary_prediction(
            logits=logits,
            apply_sigmoid=self.apply_sigmoid,
            threshold=self.threshold,
        )
        hd = _compute_hd95_matrix(
            pred=pred,
            target=label,
            include_background=self.include_background,
            directed=self.directed,
        )
        return _finite_mean(hd, fallback=self.empty_fallback)


class ChannelHD95Metric(BaseMetric):
    """
    HD95 for a selected channel index.
    """

    def __init__(
        self,
        channel_index,
        threshold=0.5,
        apply_sigmoid=True,
        include_background=True,
        directed=False,
        empty_fallback=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.channel_index = int(channel_index)
        self.threshold = threshold
        self.apply_sigmoid = apply_sigmoid
        self.include_background = include_background
        self.directed = directed
        self.empty_fallback = float(empty_fallback)

    def __call__(self, logits, label, **batch):
        pred = _prepare_binary_prediction(
            logits=logits,
            apply_sigmoid=self.apply_sigmoid,
            threshold=self.threshold,
        )
        hd = _compute_hd95_matrix(
            pred=pred,
            target=label,
            include_background=self.include_background,
            directed=self.directed,
        )
        if not (0 <= self.channel_index < hd.shape[1]):
            raise ValueError(
                f"channel_index={self.channel_index} is out of range for {hd.shape[1]} channels."
            )
        return _finite_mean(hd[:, self.channel_index], fallback=self.empty_fallback)
