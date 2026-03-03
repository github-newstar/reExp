from monai.networks.nets import SegResNet
from torch import nn


class SegResNet3D(nn.Module):
    """
    MONAI SegResNet wrapper aligned with project output contract.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        init_filters: int = 32,
        blocks_down: tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: tuple[int, ...] = (1, 1, 1),
        dropout_prob: float | None = 0.0,
    ):
        super().__init__()
        self.net = SegResNet(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
        )

    def forward(self, image, **batch):
        return {"logits": self.net(image)}

    def __str__(self):
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        return info
