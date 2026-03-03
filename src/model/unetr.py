from monai.networks.nets import UNETR
from torch import nn


class UNETR3D(nn.Module):
    """
    MONAI UNETR wrapper aligned with project output contract.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        img_size: tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: str = "instance",
        res_block: bool = True,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.net = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type=proj_type,
            norm_name=norm_name,
            res_block=res_block,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
            spatial_dims=3,
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
