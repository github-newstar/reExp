import torch
from torch import nn


def channel_shuffle_3d(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel shuffle for 3D tensors, following MedMamba-style channel mixing.
    Input shape: (B, C, D, H, W).
    """
    batch_size, channels, depth, height, width = x.shape
    if channels % groups != 0:
        raise ValueError(
            f"channels={channels} must be divisible by groups={groups} for channel shuffle."
        )

    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, depth, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels, depth, height, width)


class DIDCBlock(nn.Module):
    """
    Dilated Inception Depthwise Conv block.

    Design notes:
    - Multi-scale dilated depthwise branches follow the EMCAD multi-receptive-field idea.
    - Channel shuffle after branch concatenation follows MedMamba-style inter-branch mixing.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        reduced_channels = max(in_channels // 2, 1)

        # Step 1: projection (reduce channels by half)
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(reduced_channels),
            nn.SiLU(inplace=True),
        )

        # Step 2: parallel depthwise dilated branches
        self.branch_a = nn.Conv3d(
            reduced_channels,
            reduced_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            groups=reduced_channels,
            bias=False,
        )
        self.branch_b = nn.Conv3d(
            reduced_channels,
            reduced_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=reduced_channels,
            bias=False,
        )
        self.branch_c = nn.Conv3d(
            reduced_channels,
            reduced_channels,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=reduced_channels,
            bias=False,
        )

        # Step 3: shuffle and fuse (restore channels)
        fused_channels = reduced_channels * 3
        self.fuse = nn.Sequential(
            nn.Conv3d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )

        # Step 4: residual projection if shape mismatch
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels),
            )
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        x = self.proj(x)
        branch_a = self.branch_a(x)
        branch_b = self.branch_b(x)
        branch_c = self.branch_c(x)

        x = torch.cat([branch_a, branch_b, branch_c], dim=1)
        x = channel_shuffle_3d(x, groups=3)
        x = self.fuse(x)
        return x + residual


class _GRUFallback(nn.Module):
    """
    Fallback sequence model used when mamba_ssm is unavailable.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru(x)
        return x


class VSSBottleneck(nn.Module):
    """
    3D Vision State Space bottleneck at the lowest resolution.

    Input/Output shape: (B, C, D, H, W).
    """

    def __init__(
        self,
        channels: int,
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

        try:
            from mamba_ssm import Mamba

            self.sequence_model = Mamba(
                d_model=channels,
                d_state=mamba_state,
                d_conv=mamba_conv,
                expand=mamba_expand,
            )
        except Exception:
            self.sequence_model = _GRUFallback(d_model=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        residual = x

        # (B, C, D, H, W) -> (B, L, C), L = D*H*W
        x = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)
        x = self.norm(x)
        x = self.sequence_model(x)

        # (B, L, C) -> (B, C, D, H, W)
        x = x.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        return x + residual


class LMambaNet(nn.Module):
    """
    L-MambaNet skeleton:
    - Hybrid encoder-decoder backbone follows LBMNet-style design.
    - Encoder stages 1-3: DIDCBlock
    - Bottleneck stage 4: VSSBottleneck
    - Decoder stages 3-1: DIDCBlock
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
    ):
        super().__init__()
        c1, c2, c3, c4 = feature_channels

        # Encoder
        self.enc1 = DIDCBlock(in_channels, c1)
        self.down1 = nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc2 = DIDCBlock(c2, c2)
        self.down2 = nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc3 = DIDCBlock(c3, c3)
        self.down3 = nn.Conv3d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        # Bottleneck
        self.bottleneck = VSSBottleneck(channels=c4)

        # Decoder
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DIDCBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DIDCBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DIDCBlock(c1 + c1, c1)

        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        # Encoder path
        e1 = self.enc1(image)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        # Bottleneck
        b = self.bottleneck(self.down3(e3))

        # Decoder path with skip concatenation
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.head(d1)
        return {"logits": logits}
