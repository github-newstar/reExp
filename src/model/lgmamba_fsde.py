import torch
import torch.nn.functional as F
from torch import nn

from src.model.lgmambanet import (
    GTSMambaBottleneck,
    GTSMambaBottleneckECAMambaECAMamba,
    GTSMambaBottleneckNoECA,
    GTSMambaBottleneckPreECA,
    GTSMambaBottleneckPrePostECA,
    GTSMambaBottleneckResidualInject,
    GTSMambaBottleneckSpatialPrior,
)
from src.model.lmambanet import DIDCBlock


class ShallowDWConvResidualBlock(nn.Module):
    """
    Shallow plain feature block:
    DW-Conv3D -> BatchNorm3d -> ReLU with a simple residual connection.

    Design goal:
    avoid channel-attention/channel-shuffle bias in shallow stages so weak
    ET boundary cues are preserved as much as possible.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_proj = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.dw_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self.bn = nn.InstanceNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        residual = x
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x + residual


class LightFSDEBlock(nn.Module):
    """
    Ultra-Lightweight Frequency-Spatial Dual Enhancement block for 3D skips.

    Dual-domain idea is inspired by BraTS-UMamba / HybridMamba:
    fuse local spatial texture with frequency-domain boundary/shape cues.

    Lightweight fix:
    - Do NOT learn a full-resolution spectrum tensor.
    - Learn a tiny complex map (8x8x8) and upsample dynamically in forward.
    - This is a Low-Rank Spectrum Learning strategy for parameter efficiency.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Spatial branch: 3x3x3 depthwise conv + BN + SiLU.
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.bn = nn.InstanceNorm3d(channels)
        self.spatial_act = nn.SiLU(inplace=True)

        # Frequency branch:
        # Learn a tiny complex weight map independent of input resolution.
        # Shape: [1, 1, 8, 8, 8, 2] (real/imag), shared across batch/channels.
        weight_ri = torch.zeros(1, 1, 8, 8, 8, 2)
        weight_ri[..., 0] = 1.0
        self.small_weight = nn.Parameter(weight_ri)
        self._small_weight_bytes = (
            self.small_weight.numel() * self.small_weight.element_size()
        )
        if self._small_weight_bytes >= 10 * 1024:
            raise ValueError(
                f"small_weight consumes {self._small_weight_bytes} bytes, "
                "expected < 10KB."
            )

        # Gate: lightweight 1x1x1 conv to one attention channel, then broadcast.
        self.gate_conv = nn.Conv3d(channels, 1, kernel_size=1, bias=True)
        self.gate_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(
                f"LightFSDE channel mismatch: expected {self.channels}, got {x.shape[1]}"
            )

        # 1) Spatial path: local texture/detail extraction.
        x_spatial = self.depthwise(x)
        x_spatial = self.bn(x_spatial)
        x_spatial = self.spatial_act(x_spatial)

        # 2) Frequency path:
        # - BF16/FP16 FFT support is limited on some backends.
        #   Force frequency ops in FP32 for stability/compatibility,
        #   then cast back to the current training dtype.
        x_fft_in = x.float()
        spectrum = torch.fft.rfftn(x_fft_in, dim=(2, 3, 4))
        _, _, d, h, w_freq = spectrum.shape

        # - Low-rank spectrum learning:
        #   upsample tiny complex map to current FFT resolution.
        #   interpolate real/imag separately via channel axis.
        small_ri = self.small_weight.permute(0, 1, 5, 2, 3, 4)  # [1,1,2,8,8,8]
        small_ri = small_ri.reshape(1, 2, 8, 8, 8)  # [1,2,8,8,8]
        up_ri = F.interpolate(
            small_ri,
            size=(d, h, w_freq),
            mode="trilinear",
            align_corners=False,
        )
        up_ri = up_ri.reshape(1, 1, 2, d, h, w_freq).permute(0, 1, 3, 4, 5, 2)
        complex_weight = torch.view_as_complex(up_ri.contiguous())  # [1,1,D,H,Wf]
        modulated = spectrum * complex_weight

        # - Back to spatial domain and generate gate map.
        freq_back = torch.fft.irfftn(modulated, s=x.shape[2:], dim=(2, 3, 4))
        freq_back = freq_back.to(dtype=x.dtype)
        attention_map = self.gate_act(self.gate_conv(freq_back))

        # 3) Gated residual enhancement (frequency-guided spatial boosting).
        out = x_spatial * (1.0 + attention_map)
        return out


class LGMambaLightFSDENet(nn.Module):
    """
    LGMamba variant with LightFSDE-enhanced skip connections.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        use_channel_shuffle: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__()
        c1, c2, c3, c4 = feature_channels
        self.deep_supervision = deep_supervision
        self.use_channel_shuffle = bool(use_channel_shuffle)

        self.enc1 = DIDCBlock(
            in_channels, c1, use_channel_shuffle=self.use_channel_shuffle
        )
        self.down1 = nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc2 = DIDCBlock(c2, c2, use_channel_shuffle=self.use_channel_shuffle)
        self.down2 = nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc3 = DIDCBlock(c3, c3, use_channel_shuffle=self.use_channel_shuffle)
        self.down3 = nn.Conv3d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        # LightFSDE on skip features: frequency-guided enhancement before decoding.
        self.skip_fsde1 = LightFSDEBlock(channels=c1)
        self.skip_fsde2 = LightFSDEBlock(channels=c2)
        self.skip_fsde3 = LightFSDEBlock(channels=c3)

        self.bottleneck = GTSMambaBottleneck(
            channels=c4,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            use_channel_shuffle=self.use_channel_shuffle,
        )

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DIDCBlock(
            c3 + c3, c3, use_channel_shuffle=self.use_channel_shuffle
        )

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DIDCBlock(
            c2 + c2, c2, use_channel_shuffle=self.use_channel_shuffle
        )

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DIDCBlock(
            c1 + c1, c1, use_channel_shuffle=self.use_channel_shuffle
        )

        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)
        if self.deep_supervision:
            # Deep supervision heads on intermediate decoder stages.
            self.ds_head2 = nn.Conv3d(c2, out_channels, kernel_size=1)
            self.ds_head3 = nn.Conv3d(c3, out_channels, kernel_size=1)

    def forward(self, image: torch.Tensor, **batch) -> dict[str, torch.Tensor]:
        # Keep encoder stream unchanged; enhance only the skip features.
        e1_raw = self.enc1(image)
        e1 = self.skip_fsde1(e1_raw)

        e2_raw = self.enc2(self.down1(e1_raw))
        e2 = self.skip_fsde2(e2_raw)

        e3_raw = self.enc3(self.down2(e2_raw))
        e3 = self.skip_fsde3(e3_raw)

        b = self.bottleneck(self.down3(e3_raw))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.head(d1)
        output = {"logits": logits}

        # Emit auxiliary logits only in training mode to avoid extra
        # compute/memory during validation and deployment inference.
        if self.deep_supervision and self.training:
            target_size = logits.shape[2:]
            aux2 = self.ds_head2(d2)
            aux3 = self.ds_head3(d3)
            aux2 = F.interpolate(
                aux2, size=target_size, mode="trilinear", align_corners=False
            )
            aux3 = F.interpolate(
                aux3, size=target_size, mode="trilinear", align_corners=False
            )
            output["aux_logits"] = [aux2, aux3]

        return output


class LGMambaFSDENet(LGMambaLightFSDENet):
    """
    Backward-compatible alias.
    Existing configs using `LGMambaFSDENet` now point to the lightweight version.
    """


class LGMambaLightFSDENoShuffleNet(LGMambaLightFSDENet):
    """
    LGMamba LightFSDE variant with channel interaction disabled in:
    - DIDC blocks (encoder/decoder)
    - GTS-Mamba bottleneck fusion (ECA disabled)
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle=False,
            dynamic_eval=dynamic_eval,
        )


class LGMambaLightFSDEStage123NoECAPostECANet(LGMambaLightFSDENet):
    """
    Remove ECA in encoder/decoder stages 1/2/3, while keeping bottleneck as
    default post-ECA GTS-Mamba.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        # Keep bottleneck post-ECA behavior (use_channel_shuffle=True), then
        # explicitly disable channel interaction in enc/dec stage 1/2/3 only.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle=True,
            dynamic_eval=dynamic_eval,
        )
        c1, c2, c3, _ = feature_channels
        self.enc1 = DIDCBlock(in_channels, c1, use_channel_shuffle=False)
        self.enc2 = DIDCBlock(c2, c2, use_channel_shuffle=False)
        self.enc3 = DIDCBlock(c3, c3, use_channel_shuffle=False)
        self.dec3 = DIDCBlock(c3 + c3, c3, use_channel_shuffle=False)
        self.dec2 = DIDCBlock(c2 + c2, c2, use_channel_shuffle=False)
        self.dec1 = DIDCBlock(c1 + c1, c1, use_channel_shuffle=False)


class LGMambaLightFSDEBottleneckNoECANet(LGMambaLightFSDENet):
    """
    LGMamba LightFSDE variant where only bottleneck ECA is removed.

    Encoder/decoder channel interaction stays unchanged to isolate ablation to
    Mamba bottleneck attention.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        use_channel_shuffle: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle=use_channel_shuffle,
            dynamic_eval=dynamic_eval,
        )
        c4 = int(feature_channels[3])
        self.bottleneck = GTSMambaBottleneckNoECA(
            channels=c4,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            use_channel_shuffle=self.use_channel_shuffle,
        )


class LGMambaLightFSDEShallowPlainNet(LGMambaLightFSDENet):
    """
    LGMamba LightFSDE variant with plain shallow stages (Stage 1/2):
    - Remove channel attention/shuffle from shallow feature extraction.
    - Use only DW-Conv3D -> BN -> ReLU + residual in shallow encoder.
    - Keep deep stages and bottleneck unchanged for fair ablation.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        use_channel_shuffle_deep: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle=use_channel_shuffle_deep,
            dynamic_eval=dynamic_eval,
        )
        c1, c2, _, _ = feature_channels

        # Replace shallow DIDC blocks with plain depthwise residual blocks.
        self.enc1 = ShallowDWConvResidualBlock(in_channels, c1)
        self.enc2 = ShallowDWConvResidualBlock(c2, c2)


class LGMambaLightFSDEPreECANet(LGMambaLightFSDENet):
    """
    LGMamba LightFSDE variant with only one change:
    ECA is moved to BEFORE the tri-axis Mamba bottleneck branches.

    All other encoder/decoder/skip-FSDE components remain identical to
    LGMambaLightFSDENet for a clean ablation.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        use_channel_shuffle: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle=use_channel_shuffle,
            dynamic_eval=dynamic_eval,
        )
        c4 = int(feature_channels[3])
        self.bottleneck = GTSMambaBottleneckPreECA(
            channels=c4,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            use_channel_shuffle=self.use_channel_shuffle,
        )


class LGMambaLightFSDEPrePostECANet(LGMambaLightFSDENet):
    """
    LGMamba LightFSDE variant with ECA both before and after tri-axis Mamba.

    All other components remain identical to LGMambaLightFSDENet so the
    ablation isolates the bottleneck channel-attention placement.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        use_channel_shuffle: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle=use_channel_shuffle,
            dynamic_eval=dynamic_eval,
        )
        c4 = int(feature_channels[3])
        self.bottleneck = GTSMambaBottleneckPrePostECA(
            channels=c4,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            use_channel_shuffle=self.use_channel_shuffle,
        )


class LGMambaLightFSDESpatialPriorNet(LGMambaLightFSDENet):
    """
    LGMamba LightFSDE variant with pre-Mamba spatial local prior encoding
    (DWConv3d + LayerNorm) in bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        use_channel_shuffle: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle=use_channel_shuffle,
            dynamic_eval=dynamic_eval,
        )
        c4 = int(feature_channels[3])
        self.bottleneck = GTSMambaBottleneckSpatialPrior(
            channels=c4,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            use_channel_shuffle=self.use_channel_shuffle,
        )


class LGMambaLightFSDEResidualInjectNet(LGMambaLightFSDENet):
    """
    LGMamba LightFSDE variant with multi-scale residual injection path in
    bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        use_channel_shuffle: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle=use_channel_shuffle,
            dynamic_eval=dynamic_eval,
        )
        c4 = int(feature_channels[3])
        self.bottleneck = GTSMambaBottleneckResidualInject(
            channels=c4,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            use_channel_shuffle=self.use_channel_shuffle,
        )


class LGMambaLightFSDEShallowSkip12NoECANet(LGMambaLightFSDEShallowPlainNet):
    """
    On shallow-plain baseline, additionally remove ECA in decoder skip stages:
    dec2 and dec1.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        use_channel_shuffle_deep: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle_deep=use_channel_shuffle_deep,
            dynamic_eval=dynamic_eval,
        )
        c1, c2, _, _ = feature_channels
        self.dec2 = DIDCBlock(c2 + c2, c2, use_channel_shuffle=False)
        self.dec1 = DIDCBlock(c1 + c1, c1, use_channel_shuffle=False)


class LGMambaLightFSDEShallowECAMambaECAMambaNet(LGMambaLightFSDEShallowPlainNet):
    """
    On shallow-plain baseline, replace bottleneck with
    ECA -> Mamba -> ECA -> Mamba.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        use_channel_shuffle_deep: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle_deep=use_channel_shuffle_deep,
            dynamic_eval=dynamic_eval,
        )
        c4 = int(feature_channels[3])
        self.bottleneck = GTSMambaBottleneckECAMambaECAMamba(
            channels=c4,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            use_channel_shuffle=self.use_channel_shuffle,
        )


class LGMambaLightFSDEShallowSkip12NoECAECAMambaECAMambaNet(
    LGMambaLightFSDEShallowECAMambaECAMambaNet
):
    """
    Combined variant:
    - shallow plain encoder (no ECA in enc1/enc2)
    - dec1/dec2 remove ECA
    - bottleneck = ECA -> Mamba -> ECA -> Mamba
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        deep_supervision: bool = True,
        use_channel_shuffle_deep: bool = True,
        dynamic_eval=None,
        **_unused_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            deep_supervision=deep_supervision,
            use_channel_shuffle_deep=use_channel_shuffle_deep,
            dynamic_eval=dynamic_eval,
        )
        c1, c2, _, _ = feature_channels
        self.dec2 = DIDCBlock(c2 + c2, c2, use_channel_shuffle=False)
        self.dec1 = DIDCBlock(c1 + c1, c1, use_channel_shuffle=False)
