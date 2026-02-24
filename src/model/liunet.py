from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.model.lgmambanet import GTSMambaBottleneckECAC3TriMambaECAC3TriMamba


class _ConvBNReLU3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


def _build_activation(activation: str) -> nn.Module:
    name = activation.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation='{activation}'. Use one of: relu, relu6, gelu.")


class _DepthwiseSeparableConvBNReLU3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class _ConvBNAct3D(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        activation: str,
        groups: int = 1,
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            _build_activation(activation),
        )


class _ConvBNLinear3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        )


class _DepthwiseSeparableConvLinear3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
            ),
        )


class InvertedResidualMKIRBlock3D(nn.Module):
    """
    MKIR-style LIU block:
    1) expand channels with 1x1x1 conv
    2) multi-kernel towers in expanded space (1x1x1 conv + 3x3x3/5x5x5 depthwise conv)
    3) concatenate then linear 1x1x1 projection to low-dimensional output
    4) residual addition from low-dimensional input (with projection when needed)
    """

    def __init__(
        self,
        in_channels: int,
        branch_filters: int,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        expansion_ratio: int = 4,
        activation: str = "relu6",
    ):
        super().__init__()
        if len(kernel_sizes) != 3:
            raise ValueError("kernel_sizes must contain exactly three values for LIU-Net.")
        if any(kernel_size % 2 == 0 for kernel_size in kernel_sizes):
            raise ValueError("All kernel sizes must be odd to preserve spatial shape.")
        if expansion_ratio < 1:
            raise ValueError(f"expansion_ratio must be >= 1, got {expansion_ratio}.")

        hidden_channels = in_channels * expansion_ratio
        self.expand = _ConvBNAct3D(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            activation=activation,
        )

        towers: list[nn.Module] = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                towers.append(
                    _ConvBNAct3D(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        activation=activation,
                    )
                )
            else:
                towers.append(
                    _ConvBNAct3D(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=kernel_size,
                        activation=activation,
                        groups=hidden_channels,
                    )
                )

        self.towers = nn.ModuleList(towers)
        self.out_channels = branch_filters * len(kernel_sizes)
        self.project = _ConvBNLinear3D(
            in_channels=hidden_channels * len(kernel_sizes),
            out_channels=self.out_channels,
            kernel_size=1,
        )
        if in_channels == self.out_channels:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = _ConvBNLinear3D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(x)
        multi_scale = torch.cat([tower(expanded) for tower in self.towers], dim=1)
        compressed = self.project(multi_scale)
        return compressed + self.residual_proj(x)


class InceptionBlock3D(nn.Module):
    """
    LIU-Net Inception block with three parallel 3D convolution towers:
    1x1x1, 3x3x3 and 5x5x5, followed by channel-wise concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        branch_filters: int,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ):
        super().__init__()
        if len(kernel_sizes) != 3:
            raise ValueError("kernel_sizes must contain exactly three values for LIU-Net.")
        if any(kernel_size % 2 == 0 for kernel_size in kernel_sizes):
            raise ValueError("All kernel sizes must be odd to preserve spatial shape.")

        self.towers = nn.ModuleList(
            _ConvBNReLU3D(
                in_channels=in_channels,
                out_channels=branch_filters,
                kernel_size=kernel_size,
            )
            for kernel_size in kernel_sizes
        )
        self.out_channels = branch_filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([tower(x) for tower in self.towers], dim=1)


class InceptionDepthwiseSeparableBlock3D(nn.Module):
    """
    LIU-Net Inception block where each tower uses depthwise separable Conv3D.
    """

    def __init__(
        self,
        in_channels: int,
        branch_filters: int,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ):
        super().__init__()
        if len(kernel_sizes) != 3:
            raise ValueError("kernel_sizes must contain exactly three values for LIU-Net.")
        if any(kernel_size % 2 == 0 for kernel_size in kernel_sizes):
            raise ValueError("All kernel sizes must be odd to preserve spatial shape.")

        self.towers = nn.ModuleList(
            _DepthwiseSeparableConvBNReLU3D(
                in_channels=in_channels,
                out_channels=branch_filters,
                kernel_size=kernel_size,
            )
            for kernel_size in kernel_sizes
        )
        self.out_channels = branch_filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([tower(x) for tower in self.towers], dim=1)


class LIUNet3D(nn.Module):
    """
    Reproduction of LIU-Net (PeerJ CS 2025, cs-2787):
    - 5-level encoder-decoder.
    - Inception blocks (1x1x1, 3x3x3, 5x5x5) at each level.
    - MaxPool3D downsampling + UpSampling3D upsampling.
    - Skip connections via channel-wise concatenation.

    Notes:
    - The architecture is designed for 96x96x96 patch training/inference in this project.
    - Output spatial size equals input spatial size.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        bottleneck_branch_filters: int = 64,
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        upsample_mode: str = "nearest",
    ):
        super().__init__()
        if len(encoder_branch_filters) != 4:
            raise ValueError("LIU-Net expects four encoder levels: (4, 8, 16, 32).")

        self.upsample_mode = upsample_mode

        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        current_channels = in_channels
        skip_channels: list[int] = []
        for branch_filters in encoder_branch_filters:
            block = InceptionBlock3D(
                in_channels=current_channels,
                branch_filters=branch_filters,
                kernel_sizes=inception_kernel_sizes,
            )
            self.encoder_blocks.append(block)
            self.pool_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            current_channels = block.out_channels
            skip_channels.append(current_channels)

        self.bottleneck = InceptionBlock3D(
            in_channels=current_channels,
            branch_filters=bottleneck_branch_filters,
            kernel_sizes=inception_kernel_sizes,
        )
        current_channels = self.bottleneck.out_channels

        self.up_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for branch_filters, skip_ch in zip(
            reversed(encoder_branch_filters),
            reversed(skip_channels),
        ):
            if upsample_mode in {"trilinear"}:
                upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
            else:
                upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_layers.append(upsample)

            decoder_in_channels = current_channels + skip_ch
            decoder_block = InceptionBlock3D(
                in_channels=decoder_in_channels,
                branch_filters=branch_filters,
                kernel_sizes=inception_kernel_sizes,
            )
            self.decoder_blocks.append(decoder_block)
            current_channels = decoder_block.out_channels

        self.head = nn.Conv3d(current_channels, out_channels, kernel_size=1)

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        skip_features: list[torch.Tensor] = []
        x = image

        for encoder_block, pool in zip(self.encoder_blocks, self.pool_layers):
            x = encoder_block(x)
            skip_features.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(
            self.up_layers,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        logits = self.head(x)
        return {"logits": logits}

    def __str__(self) -> str:
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        return info


class LIUNet3DMKIR(nn.Module):
    """
    LIU-Net variant with MKIR-style inverted residual Inception blocks.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        bottleneck_branch_filters: int = 64,
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        mkir_expansion_ratio: int = 4,
        mkir_activation: str = "relu6",
        upsample_mode: str = "nearest",
    ):
        super().__init__()
        if len(encoder_branch_filters) != 4:
            raise ValueError("LIU-Net expects four encoder levels: (4, 8, 16, 32).")

        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        current_channels = in_channels
        skip_channels: list[int] = []
        for branch_filters in encoder_branch_filters:
            block = InvertedResidualMKIRBlock3D(
                in_channels=current_channels,
                branch_filters=branch_filters,
                kernel_sizes=inception_kernel_sizes,
                expansion_ratio=mkir_expansion_ratio,
                activation=mkir_activation,
            )
            self.encoder_blocks.append(block)
            self.pool_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            current_channels = block.out_channels
            skip_channels.append(current_channels)

        self.bottleneck = InvertedResidualMKIRBlock3D(
            in_channels=current_channels,
            branch_filters=bottleneck_branch_filters,
            kernel_sizes=inception_kernel_sizes,
            expansion_ratio=mkir_expansion_ratio,
            activation=mkir_activation,
        )
        current_channels = self.bottleneck.out_channels

        self.up_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for branch_filters, skip_ch in zip(
            reversed(encoder_branch_filters),
            reversed(skip_channels),
        ):
            if upsample_mode in {"trilinear"}:
                upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
            else:
                upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_layers.append(upsample)

            decoder_in_channels = current_channels + skip_ch
            decoder_block = InvertedResidualMKIRBlock3D(
                in_channels=decoder_in_channels,
                branch_filters=branch_filters,
                kernel_sizes=inception_kernel_sizes,
                expansion_ratio=mkir_expansion_ratio,
                activation=mkir_activation,
            )
            self.decoder_blocks.append(decoder_block)
            current_channels = decoder_block.out_channels

        self.head = nn.Conv3d(current_channels, out_channels, kernel_size=1)

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        skip_features: list[torch.Tensor] = []
        x = image

        for encoder_block, pool in zip(self.encoder_blocks, self.pool_layers):
            x = encoder_block(x)
            skip_features.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(
            self.up_layers,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        logits = self.head(x)
        return {"logits": logits}

    def __str__(self) -> str:
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        return info


class LIUNet3DECAC3TriMambaBottleneck(nn.Module):
    """
    Full LIU-Net variant where only the bottleneck is replaced by:
    ECA -> C/3-3TriMamba -> ECA -> C/3-3TriMamba.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        upsample_mode: str = "nearest",
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        use_channel_shuffle: bool = True,
    ):
        super().__init__()
        if len(encoder_branch_filters) != 4:
            raise ValueError("LIU-Net expects four encoder levels: (4, 8, 16, 32).")

        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        current_channels = in_channels
        skip_channels: list[int] = []
        for branch_filters in encoder_branch_filters:
            block = InceptionBlock3D(
                in_channels=current_channels,
                branch_filters=branch_filters,
                kernel_sizes=inception_kernel_sizes,
            )
            self.encoder_blocks.append(block)
            self.pool_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            current_channels = block.out_channels
            skip_channels.append(current_channels)

        self.bottleneck = GTSMambaBottleneckECAC3TriMambaECAC3TriMamba(
            channels=current_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            use_channel_shuffle=use_channel_shuffle,
        )

        self.up_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for branch_filters, skip_ch in zip(
            reversed(encoder_branch_filters),
            reversed(skip_channels),
        ):
            if upsample_mode in {"trilinear"}:
                upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
            else:
                upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_layers.append(upsample)

            decoder_in_channels = current_channels + skip_ch
            decoder_block = InceptionBlock3D(
                in_channels=decoder_in_channels,
                branch_filters=branch_filters,
                kernel_sizes=inception_kernel_sizes,
            )
            self.decoder_blocks.append(decoder_block)
            current_channels = decoder_block.out_channels

        self.head = nn.Conv3d(current_channels, out_channels, kernel_size=1)

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        skip_features: list[torch.Tensor] = []
        x = image

        for encoder_block, pool in zip(self.encoder_blocks, self.pool_layers):
            x = encoder_block(x)
            skip_features.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(
            self.up_layers,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        logits = self.head(x)
        return {"logits": logits}

    def __str__(self) -> str:
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        return info


class LIUNet3DDepthwiseSeparable(nn.Module):
    """
    LIU-Net variant where all Conv3D layers are replaced with depthwise separable Conv3D.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        bottleneck_branch_filters: int = 64,
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        upsample_mode: str = "nearest",
    ):
        super().__init__()
        if len(encoder_branch_filters) != 4:
            raise ValueError("LIU-Net expects four encoder levels: (4, 8, 16, 32).")

        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        current_channels = in_channels
        skip_channels: list[int] = []
        for branch_filters in encoder_branch_filters:
            block = InceptionDepthwiseSeparableBlock3D(
                in_channels=current_channels,
                branch_filters=branch_filters,
                kernel_sizes=inception_kernel_sizes,
            )
            self.encoder_blocks.append(block)
            self.pool_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            current_channels = block.out_channels
            skip_channels.append(current_channels)

        self.bottleneck = InceptionDepthwiseSeparableBlock3D(
            in_channels=current_channels,
            branch_filters=bottleneck_branch_filters,
            kernel_sizes=inception_kernel_sizes,
        )
        current_channels = self.bottleneck.out_channels

        self.up_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for branch_filters, skip_ch in zip(
            reversed(encoder_branch_filters),
            reversed(skip_channels),
        ):
            if upsample_mode in {"trilinear"}:
                upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
            else:
                upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_layers.append(upsample)

            decoder_in_channels = current_channels + skip_ch
            decoder_block = InceptionDepthwiseSeparableBlock3D(
                in_channels=decoder_in_channels,
                branch_filters=branch_filters,
                kernel_sizes=inception_kernel_sizes,
            )
            self.decoder_blocks.append(decoder_block)
            current_channels = decoder_block.out_channels

        # Keep the output head as a depthwise separable projection.
        self.head = _DepthwiseSeparableConvLinear3D(
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        skip_features: list[torch.Tensor] = []
        x = image

        for encoder_block, pool in zip(self.encoder_blocks, self.pool_layers):
            x = encoder_block(x)
            skip_features.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upsample, decoder_block, skip in zip(
            self.up_layers,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        logits = self.head(x)
        return {"logits": logits}

    def __str__(self) -> str:
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        return info
