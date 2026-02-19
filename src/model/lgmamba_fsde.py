import torch
from torch import nn

from src.model.lgmambanet import GTSMambaBottleneck
from src.model.lmambanet import DIDCBlock


class FSDEBlock(nn.Module):
    """
    Frequency-Spatial Dual Enhancement block for 3D skip features.

    Dual-domain idea is inspired by BraTS-UMamba / HybridMamba style designs:
    fuse local spatial texture with frequency-domain boundary/shape cues.
    """

    def __init__(self, channels: int, spatial_size: tuple[int, int, int]):
        super().__init__()
        self.channels = channels
        self.spatial_size = tuple(int(x) for x in spatial_size)

        # Spatial branch: depthwise separable conv + SiLU.
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.spatial_act = nn.SiLU(inplace=True)

        # Frequency branch: learnable complex spectrum modulation.
        d, h, w = self.spatial_size
        w_freq = w // 2 + 1

        # Store complex weight as real tensor (..., 2) for stable optimization.
        # shape matches rFFT output: [1, C, D, H, W//2+1] in complex form.
        weight_ri = torch.zeros(1, channels, d, h, w_freq, 2)
        weight_ri[..., 0] = 1.0  # real part init to identity, imag part is 0
        self.spectrum_weight = nn.Parameter(weight_ri)

        self.gate_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.gate_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(
                f"FSDE channel mismatch: expected {self.channels}, got {x.shape[1]}"
            )
        if x.shape[2:] != self.spatial_size:
            raise ValueError(
                "FSDE spatial mismatch: "
                f"expected {self.spatial_size}, got {tuple(x.shape[2:])}. "
                "Set model.input_size to match training ROI size."
            )

        # 1) Spatial path: local texture/detail extraction.
        x_spatial = self.depthwise(x)
        x_spatial = self.pointwise(x_spatial)
        x_spatial = self.spatial_act(x_spatial)

        # 2) Frequency path:
        # - rFFT to frequency domain (captures boundary/shape cues).
        spectrum = torch.fft.rfftn(x, dim=(2, 3, 4))

        # - Learnable complex modulation weight.
        complex_weight = torch.view_as_complex(self.spectrum_weight)
        modulated = spectrum * complex_weight

        # - Back to spatial domain and generate gate map.
        freq_back = torch.fft.irfftn(modulated, s=x.shape[2:], dim=(2, 3, 4))
        gate = self.gate_act(self.gate_conv(freq_back))

        # 3) Gated residual enhancement (frequency-guided spatial boosting).
        out = x_spatial * (1.0 + gate)
        return out


class LGMambaFSDENet(nn.Module):
    """
    LGMamba variant with FSDE-enhanced skip connections.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        input_size: tuple[int, int, int] = (96, 96, 96),
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        c1, c2, c3, c4 = feature_channels

        def down_size(sz: tuple[int, int, int]) -> tuple[int, int, int]:
            # Conv3d(k=3,s=2,p=1): out = floor((in + 1) / 2)
            return tuple((int(v) + 1) // 2 for v in sz)

        s1 = tuple(int(v) for v in input_size)
        s2 = down_size(s1)
        s3 = down_size(s2)

        self.enc1 = DIDCBlock(in_channels, c1)
        self.down1 = nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc2 = DIDCBlock(c2, c2)
        self.down2 = nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc3 = DIDCBlock(c3, c3)
        self.down3 = nn.Conv3d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        # FSDE on skip features: frequency-guided enhancement before decoder fusion.
        self.skip_fsde1 = FSDEBlock(channels=c1, spatial_size=s1)
        self.skip_fsde2 = FSDEBlock(channels=c2, spatial_size=s2)
        self.skip_fsde3 = FSDEBlock(channels=c3, spatial_size=s3)

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
        return {"logits": logits}
