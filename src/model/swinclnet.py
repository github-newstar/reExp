from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class _WindowMeta:
    b: int
    c: int
    d_pad: int
    h_pad: int
    w_pad: int
    d_orig: int
    h_orig: int
    w_orig: int
    window_size: int
    shift_size: int


def _window_partition_3d(
    x: torch.Tensor, window_size: int, shift_size: int = 0
) -> tuple[torch.Tensor, _WindowMeta]:
    """
    Partition (B, C, D, H, W) into 3D windows:
    returns windows (B*nW, Ws^3, C).
    """
    b, c, d, h, w = x.shape
    ws = int(window_size)
    ss = int(shift_size)
    if ws <= 0:
        raise ValueError(f"window_size must be > 0, got {ws}")
    if ss < 0 or ss >= ws:
        raise ValueError(f"shift_size must be in [0, window_size), got {ss} for ws={ws}")

    if ss > 0:
        x = torch.roll(x, shifts=(-ss, -ss, -ss), dims=(2, 3, 4))

    d_pad = (ws - d % ws) % ws
    h_pad = (ws - h % ws) % ws
    w_pad = (ws - w % ws) % ws
    if d_pad or h_pad or w_pad:
        x = F.pad(x, (0, w_pad, 0, h_pad, 0, d_pad))

    _, _, dp, hp, wp = x.shape
    x = x.view(b, c, dp // ws, ws, hp // ws, ws, wp // ws, ws)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    windows = x.view(-1, ws * ws * ws, c)
    meta = _WindowMeta(
        b=b,
        c=c,
        d_pad=dp,
        h_pad=hp,
        w_pad=wp,
        d_orig=d,
        h_orig=h,
        w_orig=w,
        window_size=ws,
        shift_size=ss,
    )
    return windows, meta


def _window_reverse_3d(windows: torch.Tensor, meta: _WindowMeta) -> torch.Tensor:
    ws = meta.window_size
    d_win = meta.d_pad // ws
    h_win = meta.h_pad // ws
    w_win = meta.w_pad // ws

    x = windows.view(meta.b, d_win, h_win, w_win, ws, ws, ws, meta.c)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    x = x.view(meta.b, meta.c, meta.d_pad, meta.h_pad, meta.w_pad)

    x = x[:, :, : meta.d_orig, : meta.h_orig, : meta.w_orig]
    if meta.shift_size > 0:
        ss = meta.shift_size
        x = torch.roll(x, shifts=(ss, ss, ss), dims=(2, 3, 4))
    return x


class _SwinWindowBlock3D(nn.Module):
    """
    3D window attention block with optional shifted windows (SW-MSA).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int = 4,
        shifted: bool = False,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.shift_size = self.window_size // 2 if shifted else 0
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows, meta = _window_partition_3d(
            x,
            window_size=self.window_size,
            shift_size=self.shift_size,
        )
        attn_in = self.norm1(windows)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        windows = windows + attn_out
        windows = windows + self.mlp(self.norm2(windows))
        return _window_reverse_3d(windows, meta)


class _SwinStage3D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        depth: int = 2,
        window_size: int = 4,
    ):
        super().__init__()
        blocks = []
        for idx in range(depth):
            blocks.append(
                _SwinWindowBlock3D(
                    channels=channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    shifted=bool(idx % 2 == 1),
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class RLKABlock3D(nn.Module):
    """
    Recursive Long Kernel Attention (paper-aligned lightweight version).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=channels,
            bias=False,
        )
        self.dw2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=7,
            stride=1,
            padding=9,
            dilation=3,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.dw2(self.dw1(x))
        gate = self.act(self.pw(attn))
        return x * (1.0 + gate)


class CSDFBlock3D(nn.Module):
    """
    Cross-Spatial-Depth Fusion used in decoder skip fusion.
    """

    def __init__(self, up_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.skip_depth = nn.Conv3d(
            skip_channels,
            skip_channels,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
            groups=skip_channels,
            bias=False,
        )
        self.skip_spatial = nn.Conv3d(
            skip_channels,
            skip_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
            groups=skip_channels,
            bias=False,
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(up_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, up_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        skip_enhanced = self.skip_depth(skip_feat) + self.skip_spatial(skip_feat)
        x = torch.cat([up_feat, skip_enhanced], dim=1)
        return self.fuse(x)


class SwinCLNet3D(nn.Module):
    """
    SwinCLNet (paper-aligned implementation):
    - Swin (W-MSA/SW-MSA) encoder
    - RLKA skip refinement
    - CSDF decoder fusion
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 320),
        num_heads: tuple[int, int, int, int, int] = (2, 2, 4, 8, 8),
        stage_depths: tuple[int, int, int, int, int] = (2, 2, 2, 2, 2),
        window_size: int = 4,
    ):
        super().__init__()
        if not (len(feature_channels) == len(num_heads) == len(stage_depths) == 5):
            raise ValueError("feature_channels/num_heads/stage_depths must all be length 5.")
        c1, c2, c3, c4, c5 = feature_channels

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(c1),
            nn.GELU(),
        )

        self.down1 = nn.Conv3d(c1, c2, kernel_size=2, stride=2, bias=False)
        self.down2 = nn.Conv3d(c2, c3, kernel_size=2, stride=2, bias=False)
        self.down3 = nn.Conv3d(c3, c4, kernel_size=2, stride=2, bias=False)
        self.down4 = nn.Conv3d(c4, c5, kernel_size=2, stride=2, bias=False)

        self.stage1 = _SwinStage3D(c1, num_heads[0], stage_depths[0], window_size)
        self.stage2 = _SwinStage3D(c2, num_heads[1], stage_depths[1], window_size)
        self.stage3 = _SwinStage3D(c3, num_heads[2], stage_depths[2], window_size)
        self.stage4 = _SwinStage3D(c4, num_heads[3], stage_depths[3], window_size)
        self.stage5 = _SwinStage3D(c5, num_heads[4], stage_depths[4], window_size)

        self.rlka1 = RLKABlock3D(c1)
        self.rlka2 = RLKABlock3D(c2)
        self.rlka3 = RLKABlock3D(c3)
        self.rlka4 = RLKABlock3D(c4)

        self.up4 = nn.ConvTranspose3d(c5, c4, kernel_size=2, stride=2)
        self.csdf4 = CSDFBlock3D(c4, c4, c4)

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.csdf3 = CSDFBlock3D(c3, c3, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.csdf2 = CSDFBlock3D(c2, c2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.csdf1 = CSDFBlock3D(c1, c1, c1)

        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        x1 = self.stage1(self.stem(image))
        x2 = self.stage2(self.down1(x1))
        x3 = self.stage3(self.down2(x2))
        x4 = self.stage4(self.down3(x3))
        x5 = self.stage5(self.down4(x4))

        s1 = self.rlka1(x1)
        s2 = self.rlka2(x2)
        s3 = self.rlka3(x3)
        s4 = self.rlka4(x4)

        d4 = self.csdf4(self.up4(x5), s4)
        d3 = self.csdf3(self.up3(d4), s3)
        d2 = self.csdf2(self.up2(d3), s2)
        d1 = self.csdf1(self.up1(d2), s1)
        logits = self.head(d1)
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
