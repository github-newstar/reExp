from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from src.model.drbd_mamba import DRBDMambaBlock3D
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


def _group_norm_half_channels(channels: int) -> int:
    if channels <= 1:
        return 1
    groups = max(1, channels // 2)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class _DoubleConv2DGNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        groups = _group_norm_half_channels(out_channels)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )


class _Down2D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AvgPool2d(kernel_size=2, stride=2),
            _DoubleConv2DGNReLU(in_channels, out_channels),
        )


class _Up2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(
                f"uC upsample expects even channels for ConvTranspose2d, got {in_channels}."
            )
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = _DoubleConv2DGNReLU(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class _USkipUNet2D(nn.Module):
    """
    Simplified 2D U-Net used by uC (no stem/out head, only down/up path).
    """

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("uC channels must contain at least two levels.")

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        in_channels = channels[0]
        for feature in channels[1:]:
            self.downs.append(_Down2D(in_channels, feature))
            in_channels = feature

        for feature in reversed(channels[:-1]):
            self.ups.append(_Up2D(in_channels, feature))
            in_channels = feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for down in self.downs:
            skips.append(x)
            x = down(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return x


class _UShapedConnectionSkip3D(nn.Module):
    """
    uC skip from the paper: run simplified 2D U-Net on stacked axial slices.
    """

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        self.unet2d = _USkipUNet2D(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x_2d = x.permute(0, 4, 1, 2, 3).reshape(b * w, c, d, h)
        x_2d = self.unet2d(x_2d)
        return x_2d.reshape(b, w, c, d, h).permute(0, 2, 3, 4, 1).contiguous()


class _DFi3D(nn.Module):
    """
    Dual Feature Integration from the paper.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False)
        self.att1 = nn.Conv3d(channels, 1, kernel_size=1, bias=True)
        self.att2 = nn.Conv3d(channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(torch.cat([x1, x2], dim=1))
        att = self.att1(x1) + self.att2(x2)
        return fused * self.sigmoid(att)


class AdditiveAttentionGate3D(nn.Module):
    """
    Additive attention gate used in Attention U-Net style skip filtering.
    x: encoder skip feature
    g: decoder gating feature (upsampled)
    """

    def __init__(
        self,
        x_channels: int,
        g_channels: int,
        inter_channels: int,
    ):
        super().__init__()
        if inter_channels <= 0:
            raise ValueError(f"inter_channels must be > 0, got {inter_channels}.")

        self.theta_x = nn.Sequential(
            nn.Conv3d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        self.phi_g = nn.Sequential(
            nn.Conv3d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        attention = self.relu(self.theta_x(x) + self.phi_g(g))
        attention = self.sigmoid(self.psi(attention))
        return x * attention


class USGLKSkip3D(nn.Module):
    """
    USG-LK skip refinement:
    1) uncertainty perceiver via 1x1x1 conv + sigmoid
    2) entropy-based uncertainty map U = -P * log(P + eps)
    3) soft sparse gate G = sigmoid((U - tau) / gamma)
    4) 7x7x7 depthwise repair on gated features
    5) residual fusion F_out = F + alpha * F_repaired
    """

    def __init__(
        self,
        channels: int,
        *,
        large_kernel_size: int = 7,
        gate_temperature: float = 0.1,
        init_tau: float = 0.5,
        init_alpha: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if large_kernel_size % 2 == 0:
            raise ValueError("large_kernel_size must be odd to preserve spatial shape.")
        if gate_temperature <= 0:
            raise ValueError(f"gate_temperature must be > 0, got {gate_temperature}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        self.uncertainty_proj = nn.Conv3d(channels, 1, kernel_size=1, bias=True)
        self.repair = nn.Conv3d(
            channels,
            channels,
            kernel_size=large_kernel_size,
            padding=large_kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.gate_temperature = float(gate_temperature)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(self.uncertainty_proj(x))
        uncertainty = -(probability * torch.log(probability + self.eps))
        gate = torch.sigmoid((uncertainty - self.tau) / self.gate_temperature)
        sparse = gate * x
        repaired = self.repair(sparse)
        return x + self.alpha * repaired


class FUESkip3D(nn.Module):
    """
    FUE skip enhancement:
    1) channel-mean + sigmoid to obtain probability map
    2) Shannon entropy uncertainty: u = -p * log(p + eps)
    3) confidence-guided enhancement: z_tilde = z + z * (1 - u)
    """

    def __init__(self, *, eps: float = 1e-6):
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(x.mean(dim=1, keepdim=True))
        uncertainty = -(probability * torch.log(probability + self.eps))
        confidence = 1.0 - uncertainty
        return x + x * confidence


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


class LIUNet3DDeepSupervision(LIUNet3D):
    """
    Original LIU-Net with deep supervision heads on decoder stages.

    Default aux heads:
    - decoder index 2: D/2 resolution
    - decoder index 1: D/4 resolution
    Aux logits are upsampled to the final logits size and returned only in training mode.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        bottleneck_branch_filters: int = 64,
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        upsample_mode: str = "nearest",
        aux_decoder_indices: Sequence[int] = (2, 1),
        deep_supervision_train_only: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_branch_filters=encoder_branch_filters,
            bottleneck_branch_filters=bottleneck_branch_filters,
            inception_kernel_sizes=inception_kernel_sizes,
            upsample_mode=upsample_mode,
        )
        self.deep_supervision_train_only = bool(deep_supervision_train_only)

        num_decoder_stages = len(self.decoder_blocks)
        self.aux_decoder_indices = tuple(int(i) for i in aux_decoder_indices)
        invalid = [i for i in self.aux_decoder_indices if i < 0 or i >= num_decoder_stages]
        if invalid:
            raise ValueError(
                f"aux_decoder_indices out of range [0, {num_decoder_stages - 1}]: {invalid}"
            )

        self.aux_heads = nn.ModuleDict()
        for idx in self.aux_decoder_indices:
            self.aux_heads[str(idx)] = nn.Conv3d(
                self.decoder_blocks[idx].out_channels,
                out_channels,
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

        decoder_features: list[torch.Tensor] = []
        for upsample, decoder_block, skip in zip(
            self.up_layers,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)
            decoder_features.append(x)

        logits = self.head(x)
        output = {"logits": logits}

        if self.deep_supervision_train_only and not self.training:
            return output

        aux_logits: list[torch.Tensor] = []
        for idx in self.aux_decoder_indices:
            aux = self.aux_heads[str(idx)](decoder_features[idx])
            aux = F.interpolate(
                aux,
                size=logits.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            aux_logits.append(aux)
        if len(aux_logits) > 0:
            output["aux_logits"] = aux_logits
        return output


class LIUNet3DUSGLK(LIUNet3D):
    """
    Original LIU-Net with USG-LK skip replacement on selected encoder levels.
    Default indices (1, 2) correspond to level-2 and level-3 skips in 1-based counting.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        bottleneck_branch_filters: int = 64,
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        upsample_mode: str = "nearest",
        usg_skip_indices: Sequence[int] = (1, 2),
        usg_kernel_size: int = 7,
        usg_gate_temperature: float = 0.1,
        usg_init_tau: float = 0.5,
        usg_init_alpha: float = 1.0,
        usg_eps: float = 1e-6,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_branch_filters=encoder_branch_filters,
            bottleneck_branch_filters=bottleneck_branch_filters,
            inception_kernel_sizes=inception_kernel_sizes,
            upsample_mode=upsample_mode,
        )

        valid_indices = set(range(len(self.encoder_blocks)))
        self.usg_skip_indices = tuple(int(i) for i in usg_skip_indices)
        invalid = [i for i in self.usg_skip_indices if i not in valid_indices]
        if invalid:
            raise ValueError(
                f"usg_skip_indices out of range {sorted(valid_indices)}: got {invalid}"
            )

        self.usg_skip_blocks = nn.ModuleDict()
        skip_channels = [block.out_channels for block in self.encoder_blocks]
        for idx in self.usg_skip_indices:
            self.usg_skip_blocks[str(idx)] = USGLKSkip3D(
                channels=skip_channels[idx],
                large_kernel_size=usg_kernel_size,
                gate_temperature=usg_gate_temperature,
                init_tau=usg_init_tau,
                init_alpha=usg_init_alpha,
                eps=usg_eps,
            )

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        skip_features: list[torch.Tensor] = []
        x = image

        for idx, (encoder_block, pool) in enumerate(zip(self.encoder_blocks, self.pool_layers)):
            x = encoder_block(x)
            skip = x
            if str(idx) in self.usg_skip_blocks:
                skip = self.usg_skip_blocks[str(idx)](skip)
            skip_features.append(skip)
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


class LIUNet3DFUE(LIUNet3D):
    """
    Original LIU-Net with FUE skip replacement on selected encoder levels.
    Default indices (1, 2) correspond to level-2 and level-3 skips in 1-based counting.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        bottleneck_branch_filters: int = 64,
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        upsample_mode: str = "nearest",
        fue_skip_indices: Sequence[int] = (1, 2),
        fue_eps: float = 1e-6,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_branch_filters=encoder_branch_filters,
            bottleneck_branch_filters=bottleneck_branch_filters,
            inception_kernel_sizes=inception_kernel_sizes,
            upsample_mode=upsample_mode,
        )

        valid_indices = set(range(len(self.encoder_blocks)))
        self.fue_skip_indices = tuple(int(i) for i in fue_skip_indices)
        invalid = [i for i in self.fue_skip_indices if i not in valid_indices]
        if invalid:
            raise ValueError(
                f"fue_skip_indices out of range {sorted(valid_indices)}: got {invalid}"
            )

        self.fue_skip_blocks = nn.ModuleDict()
        for idx in self.fue_skip_indices:
            self.fue_skip_blocks[str(idx)] = FUESkip3D(eps=fue_eps)

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        skip_features: list[torch.Tensor] = []
        x = image

        for idx, (encoder_block, pool) in enumerate(zip(self.encoder_blocks, self.pool_layers)):
            x = encoder_block(x)
            skip = x
            if str(idx) in self.fue_skip_blocks:
                skip = self.fue_skip_blocks[str(idx)](skip)
            skip_features.append(skip)
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


class LIUNet3DAttentionGate(nn.Module):
    """
    LIU-Net variant that inserts additive attention gates on all skip connections.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        bottleneck_branch_filters: int = 64,
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        upsample_mode: str = "nearest",
        ag_inter_ratio: float = 0.5,
    ):
        super().__init__()
        if len(encoder_branch_filters) != 4:
            raise ValueError("LIU-Net expects four encoder levels: (4, 8, 16, 32).")
        if not (0.0 < ag_inter_ratio <= 1.0):
            raise ValueError(f"ag_inter_ratio must be in (0,1], got {ag_inter_ratio}.")

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
        self.attention_gates = nn.ModuleList()
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

            inter_ch = max(1, int(round(skip_ch * ag_inter_ratio)))
            self.attention_gates.append(
                AdditiveAttentionGate3D(
                    x_channels=skip_ch,
                    g_channels=current_channels,
                    inter_channels=inter_ch,
                )
            )

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

        for upsample, attention_gate, decoder_block, skip in zip(
            self.up_layers,
            self.attention_gates,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x)
            skip = attention_gate(skip, x)
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


class LIUNet3DAddSkip(nn.Module):
    """
    LIU-Net variant using element-wise-add skip fusion.
    Decoder features are projected by 1x1x1 conv before addition.
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
        self.up_proj_layers = nn.ModuleList()
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
            self.up_proj_layers.append(
                nn.Conv3d(current_channels, skip_ch, kernel_size=1, bias=False)
            )

            decoder_block = InceptionBlock3D(
                in_channels=skip_ch,
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

        for upsample, up_proj, decoder_block, skip in zip(
            self.up_layers,
            self.up_proj_layers,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x)
            x = up_proj(x)
            x = x + skip
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


class LIUNet3DDRBDMamba(nn.Module):
    """
    Original LIU-Net backbone + DRBD-Mamba on:
    - bottleneck
    - selected skip levels (default: deepest skip only)
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        bottleneck_branch_filters: int = 64,
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        upsample_mode: str = "nearest",
        drbd_skip_indices: Sequence[int] = (3,),
        drbd_mamba_state: int = 16,
        drbd_mamba_conv: int = 4,
        drbd_mamba_expand: int = 2,
        drbd_sequence_impl: str = "mamba1",
        drbd_use_quantizer: bool = True,
        drbd_num_embeddings: int = 256,
        drbd_ema_decay: float = 0.99,
        drbd_commitment_weight: float = 0.25,
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

        valid_indices = set(range(len(skip_channels)))
        self.drbd_skip_indices = tuple(int(i) for i in drbd_skip_indices)
        invalid = [i for i in self.drbd_skip_indices if i not in valid_indices]
        if invalid:
            raise ValueError(
                f"drbd_skip_indices out of range {sorted(valid_indices)}: got {invalid}"
            )

        self.skip_drbd_blocks = nn.ModuleDict()
        for idx in self.drbd_skip_indices:
            self.skip_drbd_blocks[str(idx)] = DRBDMambaBlock3D(
                channels=skip_channels[idx],
                mamba_state=drbd_mamba_state,
                mamba_conv=drbd_mamba_conv,
                mamba_expand=drbd_mamba_expand,
                sequence_impl=drbd_sequence_impl,
                use_quantizer=drbd_use_quantizer,
                num_embeddings=drbd_num_embeddings,
                ema_decay=drbd_ema_decay,
                commitment_weight=drbd_commitment_weight,
            )

        # Keep original bottleneck width (64 per tower => 192 channels) via
        # point-wise projection before DRBD-Mamba.
        bottleneck_channels = bottleneck_branch_filters * len(inception_kernel_sizes)
        self.bottleneck_proj = _ConvBNReLU3D(
            in_channels=current_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
        )
        self.bottleneck = DRBDMambaBlock3D(
            channels=bottleneck_channels,
            mamba_state=drbd_mamba_state,
            mamba_conv=drbd_mamba_conv,
            mamba_expand=drbd_mamba_expand,
            sequence_impl=drbd_sequence_impl,
            use_quantizer=drbd_use_quantizer,
            num_embeddings=drbd_num_embeddings,
            ema_decay=drbd_ema_decay,
            commitment_weight=drbd_commitment_weight,
        )
        current_channels = bottleneck_channels

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
        drbd_commit_terms: list[torch.Tensor] = []
        drbd_perplexity_stats: dict[str, torch.Tensor] = {}

        for idx, (encoder_block, pool) in enumerate(zip(self.encoder_blocks, self.pool_layers)):
            x = encoder_block(x)
            skip = x
            if str(idx) in self.skip_drbd_blocks:
                skip, skip_stats = self.skip_drbd_blocks[str(idx)](skip)
                drbd_commit_terms.append(skip_stats["commitment_loss"])
                drbd_perplexity_stats[f"drbd_skip{idx + 1}_perplexity"] = skip_stats[
                    "perplexity"
                ].detach()
            skip_features.append(skip)
            x = pool(x)

        x = self.bottleneck_proj(x)
        x, bottleneck_stats = self.bottleneck(x)
        drbd_commit_terms.append(bottleneck_stats["commitment_loss"])
        drbd_perplexity_stats["drbd_bottleneck_perplexity"] = bottleneck_stats[
            "perplexity"
        ].detach()

        for upsample, decoder_block, skip in zip(
            self.up_layers,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        logits = self.head(x)
        output = {"logits": logits}
        if self.training and len(drbd_commit_terms) > 0:
            output["drbd_commit_loss"] = torch.stack(drbd_commit_terms).sum()
            output.update(drbd_perplexity_stats)
        return output

    def __str__(self) -> str:
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        return info


class LIUNet3DDRBDMamba2(LIUNet3DDRBDMamba):
    """
    Original LIU-Net backbone + DRBD-Mamba with Mamba2 sequence operator.
    """

    def __init__(self, *args, **kwargs):
        kwargs["drbd_sequence_impl"] = "mamba2"
        super().__init__(*args, **kwargs)


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


class LIUNet3DMKIRAddSkip(nn.Module):
    """
    LIU-Net MKIR variant with element-wise-add skip fusion.
    Decoder features are projected by 1x1x1 conv before addition.
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
        self.up_proj_layers = nn.ModuleList()
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
            self.up_proj_layers.append(
                nn.Conv3d(current_channels, skip_ch, kernel_size=1, bias=False)
            )

            decoder_block = InvertedResidualMKIRBlock3D(
                in_channels=skip_ch,
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

        for upsample, up_proj, decoder_block, skip in zip(
            self.up_layers,
            self.up_proj_layers,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x)
            x = up_proj(x)
            x = x + skip
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


class LIUNet3DMKIRDRBDMamba(nn.Module):
    """
    LIU-Net MKIR variant with DRBD-Mamba applied to:
    - bottleneck latent feature
    - selected skip connection levels (default: deepest skip only)
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        mkir_expansion_ratio: int = 4,
        mkir_activation: str = "relu6",
        upsample_mode: str = "nearest",
        drbd_skip_indices: Sequence[int] = (3,),
        drbd_mamba_state: int = 16,
        drbd_mamba_conv: int = 4,
        drbd_mamba_expand: int = 2,
        drbd_use_quantizer: bool = True,
        drbd_num_embeddings: int = 256,
        drbd_ema_decay: float = 0.99,
        drbd_commitment_weight: float = 0.25,
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

        self.bottleneck = DRBDMambaBlock3D(
            channels=current_channels,
            mamba_state=drbd_mamba_state,
            mamba_conv=drbd_mamba_conv,
            mamba_expand=drbd_mamba_expand,
            use_quantizer=drbd_use_quantizer,
            num_embeddings=drbd_num_embeddings,
            ema_decay=drbd_ema_decay,
            commitment_weight=drbd_commitment_weight,
        )

        valid_indices = set(range(len(skip_channels)))
        self.drbd_skip_indices = tuple(int(i) for i in drbd_skip_indices)
        invalid = [i for i in self.drbd_skip_indices if i not in valid_indices]
        if invalid:
            raise ValueError(
                f"drbd_skip_indices out of range {sorted(valid_indices)}: got {invalid}"
            )

        self.skip_drbd_blocks = nn.ModuleDict()
        for idx in self.drbd_skip_indices:
            self.skip_drbd_blocks[str(idx)] = DRBDMambaBlock3D(
                channels=skip_channels[idx],
                mamba_state=drbd_mamba_state,
                mamba_conv=drbd_mamba_conv,
                mamba_expand=drbd_mamba_expand,
                use_quantizer=drbd_use_quantizer,
                num_embeddings=drbd_num_embeddings,
                ema_decay=drbd_ema_decay,
                commitment_weight=drbd_commitment_weight,
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
        drbd_commit_terms: list[torch.Tensor] = []
        drbd_perplexity_stats: dict[str, torch.Tensor] = {}

        for idx, (encoder_block, pool) in enumerate(zip(self.encoder_blocks, self.pool_layers)):
            x = encoder_block(x)
            skip = x
            if str(idx) in self.skip_drbd_blocks:
                skip, skip_stats = self.skip_drbd_blocks[str(idx)](skip)
                drbd_commit_terms.append(skip_stats["commitment_loss"])
                drbd_perplexity_stats[f"drbd_skip{idx + 1}_perplexity"] = skip_stats[
                    "perplexity"
                ].detach()
            skip_features.append(skip)
            x = pool(x)

        x, bottleneck_stats = self.bottleneck(x)
        drbd_commit_terms.append(bottleneck_stats["commitment_loss"])
        drbd_perplexity_stats["drbd_bottleneck_perplexity"] = bottleneck_stats[
            "perplexity"
        ].detach()

        for upsample, decoder_block, skip in zip(
            self.up_layers,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        logits = self.head(x)
        output = {"logits": logits}
        if self.training and len(drbd_commit_terms) > 0:
            drbd_commit_loss = torch.stack(drbd_commit_terms).sum()
            output["drbd_commit_loss"] = drbd_commit_loss
            output.update(drbd_perplexity_stats)
        return output

    def __str__(self) -> str:
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        return info


class LIUNet3DMKIRECAC3TriMambaBottleneck(nn.Module):
    """
    LIU-Net MKIR variant where only bottleneck is replaced by:
    ECA -> C/3-3TriMamba -> ECA -> C/3-3TriMamba.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_branch_filters: Sequence[int] = (4, 8, 16, 32),
        inception_kernel_sizes: Sequence[int] = (1, 3, 5),
        mkir_expansion_ratio: int = 4,
        mkir_activation: str = "relu6",
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


class LIUNet3DUCDFi(nn.Module):
    """
    LIU-Net variant following uC-3DU-Net:
    - replace skip1/2/3 with uC modules
    - use DFi on the first three decoder fusion stages
    - keep the final decoder stage as regular concat + Inception decoding
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
        bottleneck_channels = self.bottleneck.out_channels
        current_channels = bottleneck_channels

        self.uc_skip1 = _UShapedConnectionSkip3D(
            channels=(
                skip_channels[0],
                skip_channels[1],
                skip_channels[2],
                skip_channels[3],
                bottleneck_channels,
            )
        )
        self.uc_skip2 = _UShapedConnectionSkip3D(
            channels=(
                skip_channels[1],
                skip_channels[2],
                skip_channels[3],
                bottleneck_channels,
            )
        )
        self.uc_skip3 = _UShapedConnectionSkip3D(
            channels=(skip_channels[2], skip_channels[3], bottleneck_channels)
        )

        self.up_layers = nn.ModuleList()
        self.up_proj_layers = nn.ModuleList()
        self.dfi_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for stage_idx, (branch_filters, skip_ch) in enumerate(
            zip(reversed(encoder_branch_filters), reversed(skip_channels))
        ):
            if upsample_mode in {"trilinear"}:
                upsample = nn.Upsample(
                    scale_factor=2, mode=upsample_mode, align_corners=False
                )
            else:
                upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_layers.append(upsample)

            if stage_idx < 3:
                self.up_proj_layers.append(
                    nn.Conv3d(current_channels, skip_ch, kernel_size=1, bias=False)
                )
                self.dfi_layers.append(_DFi3D(skip_ch))
                decoder_in_channels = skip_ch
            else:
                self.up_proj_layers.append(nn.Identity())
                decoder_in_channels = current_channels + skip_ch

            decoder_block = InceptionBlock3D(
                in_channels=decoder_in_channels,
                branch_filters=branch_filters,
                kernel_sizes=inception_kernel_sizes,
            )
            self.decoder_blocks.append(decoder_block)
            current_channels = decoder_block.out_channels

        self.head = nn.Conv3d(current_channels, out_channels, kernel_size=1)

    @staticmethod
    def _resize_to_match(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] != target.shape[-3:]:
            return F.interpolate(
                x, size=target.shape[-3:], mode="trilinear", align_corners=False
            )
        return x

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        skip_features: list[torch.Tensor] = []
        x = image

        for encoder_block, pool in zip(self.encoder_blocks, self.pool_layers):
            x = encoder_block(x)
            skip_features.append(x)
            x = pool(x)

        # Replace skip1/2/3 by uC skip features; keep the deepest skip unchanged.
        skip_features[0] = self.uc_skip1(skip_features[0])
        skip_features[1] = self.uc_skip2(skip_features[1])
        skip_features[2] = self.uc_skip3(skip_features[2])

        x = self.bottleneck(x)

        for stage_idx, (upsample, up_proj, decoder_block, skip) in enumerate(
            zip(
                self.up_layers,
                self.up_proj_layers,
                self.decoder_blocks,
                reversed(skip_features),
            )
        ):
            x = upsample(x)
            x = self._resize_to_match(x, skip)

            if stage_idx < 3:
                x = up_proj(x)
                x = self._resize_to_match(x, skip)
                x = self.dfi_layers[stage_idx](x, skip)
            else:
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
