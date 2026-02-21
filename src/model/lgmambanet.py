import torch
from torch import nn

from src.model.lmambanet import DIDCBlock, ECABlock3D


class GTSMambaBottleneck(nn.Module):
    """
    Grouped Tri-Axis Shuffling Mamba bottleneck for 3D volumes.

    The block is lightweight by splitting channels into 3 groups and applying
    axis-specific Mamba scans:
    - Branch 1: Intra-slice scan on each axial plane (captures planar features).
    - Branch 2: Depth-axis scan across slices (captures volumetric continuity).
    - Branch 3: Global flatten scan (captures long-range global dependencies).

    Then branch outputs are concatenated, ECA reweights channels for soft
    cross-channel interaction, fused by 1x1x1 conv, and added to a residual.
    """

    def __init__(
        self,
        channels: int,
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        use_channel_shuffle: bool = True,
    ):
        super().__init__()
        if channels < 3:
            raise ValueError(f"channels must be >= 3 for 3-way grouping, got {channels}")
        self.use_channel_shuffle = bool(use_channel_shuffle)
        from mamba_ssm import Mamba

        # Make branch channels equal for stable grouping/shuffle if C % 3 != 0.
        branch_channels = (channels + 2) // 3
        grouped_channels = branch_channels * 3
        self.use_group_proj = grouped_channels != channels
        self.group_proj = (
            nn.Conv3d(channels, grouped_channels, kernel_size=1, bias=False)
            if self.use_group_proj
            else nn.Identity()
        )

        self.norm1 = nn.LayerNorm(branch_channels)
        self.norm2 = nn.LayerNorm(branch_channels)
        self.norm3 = nn.LayerNorm(branch_channels)

        self.mamba1 = Mamba(
            d_model=branch_channels,
            d_state=mamba_state,
            d_conv=mamba_conv,
            expand=mamba_expand,
        )
        self.mamba2 = Mamba(
            d_model=branch_channels,
            d_state=mamba_state,
            d_conv=mamba_conv,
            expand=mamba_expand,
        )
        self.mamba3 = Mamba(
            d_model=branch_channels,
            d_state=mamba_state,
            d_conv=mamba_conv,
            expand=mamba_expand,
        )

        self.channel_interaction = (
            ECABlock3D(channels=grouped_channels)
            if self.use_channel_shuffle
            else nn.Identity()
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(grouped_channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.SiLU(inplace=True),
        )

    def _branch_intra_slice(self, x1: torch.Tensor) -> torch.Tensor:
        """
        Branch 1:
        For each depth slice, scan tokens on the HxW plane (intra-slice planar context).
        Input/Output: (B, Cg, D, H, W)
        """
        b, cg, d, h, w = x1.shape
        seq = x1.permute(0, 2, 3, 4, 1).reshape(b * d, h * w, cg)
        seq = self.mamba1(self.norm1(seq))
        return seq.view(b, d, h, w, cg).permute(0, 4, 1, 2, 3).contiguous()

    def _branch_depth_scan(self, x2: torch.Tensor) -> torch.Tensor:
        """
        Branch 2:
        For each (H, W) location, scan sequence along depth D (inter-slice continuity).
        Input/Output: (B, Cg, D, H, W)
        """
        b, cg, d, h, w = x2.shape
        seq = x2.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, cg)
        seq = self.mamba2(self.norm2(seq))
        return seq.view(b, h, w, d, cg).permute(0, 4, 3, 1, 2).contiguous()

    def _branch_global(self, x3: torch.Tensor) -> torch.Tensor:
        """
        Branch 3:
        Flatten the whole 3D volume as one global sequence.
        Input/Output: (B, Cg, D, H, W)
        """
        b, cg, d, h, w = x3.shape
        seq = x3.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, cg)
        seq = self.mamba3(self.norm3(seq))
        return seq.view(b, d, h, w, cg).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        grouped = self.group_proj(x)

        # Group channels into 3 chunks for tri-axis scanning.
        x1, x2, x3 = torch.chunk(grouped, chunks=3, dim=1)

        out1 = self._branch_intra_slice(x1)
        out2 = self._branch_depth_scan(x2)
        out3 = self._branch_global(x3)

        # Concatenate tri-axis outputs and apply ECA channel interaction
        # before pointwise fusion.
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.channel_interaction(out)
        out = self.fuse(out)
        return out + residual


class GTSMambaBottleneckPreECA(GTSMambaBottleneck):
    """
    Pre-ECA variant of GTS-Mamba bottleneck.

    Only architectural difference vs. GTSMambaBottleneck:
    - Move ECA channel interaction BEFORE tri-axis Mamba branches
    - Disable post-branch ECA after concat

    This keeps parameter/FLOP changes minimal and isolates the ablation to
    ECA placement around the bottleneck.
    """

    def __init__(
        self,
        channels: int,
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        use_channel_shuffle: bool = True,
    ):
        super().__init__(
            channels=channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            use_channel_shuffle=use_channel_shuffle,
        )
        # Reuse the exact same ECA module but apply it before branching.
        self.pre_channel_interaction = self.channel_interaction
        self.channel_interaction = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        grouped = self.group_proj(x)
        grouped = self.pre_channel_interaction(grouped)

        x1, x2, x3 = torch.chunk(grouped, chunks=3, dim=1)

        out1 = self._branch_intra_slice(x1)
        out2 = self._branch_depth_scan(x2)
        out3 = self._branch_global(x3)

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fuse(out)
        return out + residual


class GTSMambaBottleneckPrePostECA(GTSMambaBottleneck):
    """
    Pre+Post-ECA variant of GTS-Mamba bottleneck.

    Relative to GTSMambaBottleneck:
    - Add one ECA before tri-axis Mamba branches.
    - Keep original post-branch ECA before fuse.
    """

    def __init__(
        self,
        channels: int,
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        use_channel_shuffle: bool = True,
    ):
        super().__init__(
            channels=channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            use_channel_shuffle=use_channel_shuffle,
        )
        grouped_channels = (
            self.group_proj.out_channels
            if isinstance(self.group_proj, nn.Conv3d)
            else channels
        )
        self.pre_channel_interaction = (
            ECABlock3D(channels=grouped_channels)
            if self.use_channel_shuffle
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        grouped = self.group_proj(x)
        grouped = self.pre_channel_interaction(grouped)

        x1, x2, x3 = torch.chunk(grouped, chunks=3, dim=1)

        out1 = self._branch_intra_slice(x1)
        out2 = self._branch_depth_scan(x2)
        out3 = self._branch_global(x3)

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.channel_interaction(out)
        out = self.fuse(out)
        return out + residual


class LGMambaNet(nn.Module):
    """
    L-MambaNet variant with GTS-Mamba bottleneck (LGMambaNet).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        c1, c2, c3, c4 = feature_channels

        self.enc1 = DIDCBlock(in_channels, c1)
        self.down1 = nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc2 = DIDCBlock(c2, c2)
        self.down2 = nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc3 = DIDCBlock(c3, c3)
        self.down3 = nn.Conv3d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        self.bottleneck = GTSMambaBottleneck(
            channels=c4,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
        )

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DIDCBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DIDCBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DIDCBlock(c1 + c1, c1)

        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        e1 = self.enc1(image)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.head(d1)
        return {"logits": logits}
