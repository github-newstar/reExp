import torch
from torch import nn
import torch.nn.functional as F


class _ConvINLReLU(nn.Module):
    """
    Conv3d -> InstanceNorm3d -> LeakyReLU block.
    """

    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 1e-2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _NoNewNetStage(nn.Module):
    """
    Two Conv-IN-LReLU sequences per resolution stage.
    """

    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 1e-2):
        super().__init__()
        self.block = nn.Sequential(
            _ConvINLReLU(in_channels, out_channels, negative_slope=negative_slope),
            _ConvINLReLU(out_channels, out_channels, negative_slope=negative_slope),
        )

    def forward(self, x):
        return self.block(x)


class NoNewNet(nn.Module):
    """
    BraTS classic No New-Net style 3D U-Net baseline:
    - output stride 1
    - 2x (Conv + InstanceNorm + LeakyReLU) per resolution
    - 4 downsampling operations
    - 30 initial feature maps, doubling each downsampling (30->60->120->240->480)
    - mirrored decoder
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_features: int = 30,
        negative_slope: float = 1e-2,
    ):
        super().__init__()
        f0 = int(base_features)
        f1 = f0 * 2
        f2 = f1 * 2
        f3 = f2 * 2
        f4 = f3 * 2

        self.enc0 = _NoNewNetStage(in_channels, f0, negative_slope=negative_slope)
        self.enc1 = _NoNewNetStage(f0, f1, negative_slope=negative_slope)
        self.enc2 = _NoNewNetStage(f1, f2, negative_slope=negative_slope)
        self.enc3 = _NoNewNetStage(f2, f3, negative_slope=negative_slope)
        self.bottleneck = _NoNewNetStage(f3, f4, negative_slope=negative_slope)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.up3 = nn.ConvTranspose3d(f4, f3, kernel_size=2, stride=2)
        self.dec3 = _NoNewNetStage(f3 + f3, f3, negative_slope=negative_slope)
        self.up2 = nn.ConvTranspose3d(f3, f2, kernel_size=2, stride=2)
        self.dec2 = _NoNewNetStage(f2 + f2, f2, negative_slope=negative_slope)
        self.up1 = nn.ConvTranspose3d(f2, f1, kernel_size=2, stride=2)
        self.dec1 = _NoNewNetStage(f1 + f1, f1, negative_slope=negative_slope)
        self.up0 = nn.ConvTranspose3d(f1, f0, kernel_size=2, stride=2)
        self.dec0 = _NoNewNetStage(f0 + f0, f0, negative_slope=negative_slope)

        self.head = nn.Conv3d(f0, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(module.weight, a=1e-2, mode="fan_out", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _concat_with_skip(upsampled, skip):
        if upsampled.shape[-3:] != skip.shape[-3:]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        return torch.cat([skip, upsampled], dim=1)

    def forward(self, image, **batch):
        x0 = self.enc0(image)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))

        y3 = self.up3(xb)
        y3 = self.dec3(self._concat_with_skip(y3, x3))
        y2 = self.up2(y3)
        y2 = self.dec2(self._concat_with_skip(y2, x2))
        y1 = self.up1(y2)
        y1 = self.dec1(self._concat_with_skip(y1, x1))
        y0 = self.up0(y1)
        y0 = self.dec0(self._concat_with_skip(y0, x0))

        return {"logits": self.head(y0)}

    def __str__(self):
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        return info
