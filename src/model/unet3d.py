from monai.networks.nets import UNet
from torch import nn


class UNet3D(nn.Module):
    """
    MONAI UNet wrapper aligned with project output contract.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
        strides: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_res_units: int = 0,
    ):
        super().__init__()
        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
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
