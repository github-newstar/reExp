from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from torch import nn


def _build_mamba(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    num_slices: int | None = None,
) -> nn.Module:
    """
    Build Mamba block with best-effort compatibility across mamba-ssm versions.
    """
    try:
        from mamba_ssm import Mamba
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "SegMamba3D requires mamba-ssm. "
            "Please install mamba-ssm in the current environment."
        ) from exc

    kwargs = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    # Prefer official SegMamba arguments when the installed Mamba supports them.
    try:
        return Mamba(**kwargs, bimamba_type="v3", nslices=num_slices)
    except TypeError:
        return Mamba(**kwargs)


class _MlpChannel(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _GSC(nn.Module):
    """
    Global-Spatial Context module from official SegMamba implementation.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm3d(channels)
        self.act = nn.ReLU(inplace=True)

        self.proj2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.act2 = nn.ReLU(inplace=True)

        self.proj3 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.InstanceNorm3d(channels)
        self.act3 = nn.ReLU(inplace=True)

        self.proj4 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.norm4 = nn.InstanceNorm3d(channels)
        self.act4 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x1 = self.act(self.norm(self.proj(x)))
        x1 = self.act2(self.norm2(self.proj2(x1)))

        x2 = self.act3(self.norm3(self.proj3(x)))

        out = x1 + x2
        out = self.act4(self.norm4(self.proj4(out)))
        return out + residual


class _MambaLayer3D(nn.Module):
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_slices: int | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.norm = nn.LayerNorm(channels)
        self.mamba = _build_mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_slices=num_slices,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        if c != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {c}")
        residual = x
        n_tokens = x.shape[2:].numel()
        spatial = x.shape[2:]
        seq = x.reshape(b, c, n_tokens).transpose(-1, -2)
        seq = self.norm(seq)
        seq = self.mamba(seq)
        out = seq.transpose(-1, -2).reshape(b, c, *spatial)
        return out + residual


class _SegMambaEncoder(nn.Module):
    """
    SegMamba encoder that follows the official repo structure.
    """

    def __init__(
        self,
        in_channels: int,
        depths: Sequence[int],
        dims: Sequence[int],
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("SegMamba encoder expects 4 stages for depths and dims.")

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(nn.Conv3d(in_channels, dims[0], kernel_size=7, stride=2, padding=3))
        )
        for idx in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.InstanceNorm3d(dims[idx]),
                    nn.Conv3d(dims[idx], dims[idx + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        # Same defaults as official SegMamba repo.
        num_slices = [64, 32, 16, 8]
        for idx in range(4):
            self.gscs.append(_GSC(dims[idx]))
            blocks = [_MambaLayer3D(dims[idx], num_slices=num_slices[idx]) for _ in range(depths[idx])]
            self.stages.append(nn.Sequential(*blocks))

        self.norms = nn.ModuleList([nn.InstanceNorm3d(d) for d in dims])
        self.mlps = nn.ModuleList([_MlpChannel(d, 2 * d) for d in dims])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outs: list[torch.Tensor] = []
        for idx in range(4):
            x = self.downsample_layers[idx](x)
            x = self.gscs[idx](x)
            x = self.stages[idx](x)
            x = self.norms[idx](x)
            x = self.mlps[idx](x)
            outs.append(x)
        return tuple(outs)  # type: ignore[return-value]


class SegMamba3D(nn.Module):
    """
    SegMamba implementation adapted from official code:
    https://github.com/ge-xing/SegMamba
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        feat_size: tuple[int, int, int, int] = (48, 96, 192, 384),
        hidden_size: int = 768,
        norm_name: str = "instance",
        res_block: bool = True,
        spatial_dims: int = 3,
    ):
        super().__init__()
        self.encoder_backbone = _SegMambaEncoder(
            in_channels=in_channels,
            depths=depths,
            dims=feat_size,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=out_channels,
        )

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        o1, o2, o3, o4 = self.encoder_backbone(image)

        enc1 = self.encoder1(image)
        enc2 = self.encoder2(o1)
        enc3 = self.encoder3(o2)
        enc4 = self.encoder4(o3)
        hidden = self.encoder5(o4)

        dec3 = self.decoder5(hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        logits = self.out(out)
        return {"logits": logits}

    def __str__(self):
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        return info
