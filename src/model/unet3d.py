from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from monai.networks.layers import SkipConnection
from monai.networks.nets import UNet
from torch import nn


class UNet3D(nn.Module):
    """
    MONAI UNet wrapper aligned with project output contract.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
        strides: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_res_units: int = 0,
    ):
        super().__init__()
        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
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


class FullDualBandFSDEBlock3D(nn.Module):
    """
    Full Dual-Band FSDE block.

    - Spatial branch: DWConv3d(3x3x3) (+ optional IN + SiLU)
    - Frequency branch: per-channel full-resolution LF/HF masks in rFFT space
    - Dual-band gate: g from [Y_L; Y_H]
    - Gain injection: Out = S * (1 + G)
    """

    def __init__(
        self,
        channels: int,
        fft_shape: Sequence[int],  # (D_f, H_f, W_f) in rFFT space
        alpha: float = 0.2,
        learnable_alpha: bool = False,
        alpha_max: float = 0.3,
        mix_gate_channels: int = 1,  # 1: spatial gate, C: spatial+channel gate
        gain_channels: int = 1,  # 1: spatial gain, C: spatial+channel gain
        use_norm_act: bool = True,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if len(fft_shape) != 3:
            raise ValueError(f"fft_shape must be (D_f,H_f,W_f), got {fft_shape}")
        if any(int(v) <= 0 for v in fft_shape):
            raise ValueError(f"fft_shape must be positive, got {fft_shape}")
        if mix_gate_channels not in (1, channels):
            raise ValueError(
                f"mix_gate_channels must be 1 or channels({channels}), got {mix_gate_channels}"
            )
        if gain_channels not in (1, channels):
            raise ValueError(
                f"gain_channels must be 1 or channels({channels}), got {gain_channels}"
            )

        d_f, h_f, w_f = (int(v) for v in fft_shape)
        self.channels = int(channels)
        self.fft_shape = (d_f, h_f, w_f)
        self.alpha_max = float(alpha_max)

        self.spatial_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.spatial_norm = nn.InstanceNorm3d(channels) if use_norm_act else nn.Identity()
        self.spatial_act = nn.SiLU(inplace=True) if use_norm_act else nn.Identity()

        # Full + channel-wise masks for LF and HF.
        self.weight_lf = nn.Parameter(torch.zeros(1, channels, d_f, h_f, w_f))
        self.weight_hf = nn.Parameter(torch.zeros(1, channels, d_f, h_f, w_f))

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

        self.mix_gate = nn.Conv3d(channels * 2, mix_gate_channels, kernel_size=1, bias=True)
        self.gain_proj = nn.Conv3d(channels, gain_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def _alpha_value(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        alpha = self.alpha.to(dtype=dtype, device=device)
        return torch.clamp(alpha, 0.0, self.alpha_max)

    @staticmethod
    def _maybe_resize_weight(weight: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if tuple(weight.shape[-3:]) == tuple(target_shape):
            return weight
        return F.interpolate(weight, size=target_shape, mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(
                f"FullDualBandFSDEBlock3D channel mismatch: expected {self.channels}, got {x.shape[1]}"
            )

        # 1) Spatial branch
        s = self.spatial_act(self.spatial_norm(self.spatial_dw(x)))

        # 2) Frequency dual-band branch
        x_fft = x.float()
        f = torch.fft.rfftn(x_fft, dim=(2, 3, 4))
        _, _, d_f, h_f, w_f = f.shape
        current_fft_shape = (d_f, h_f, w_f)

        w_lf = self._maybe_resize_weight(self.weight_lf, current_fft_shape)
        w_hf = self._maybe_resize_weight(self.weight_hf, current_fft_shape)
        alpha = self._alpha_value(dtype=w_lf.dtype, device=w_lf.device)
        m_lf = 1.0 + alpha * torch.tanh(w_lf)
        m_hf = 1.0 + alpha * torch.tanh(w_hf)

        f_lf = f * m_lf
        f_hf = f * m_hf

        y_lf = torch.fft.irfftn(f_lf, s=x.shape[2:], dim=(2, 3, 4)).to(dtype=x.dtype)
        y_hf = torch.fft.irfftn(f_hf, s=x.shape[2:], dim=(2, 3, 4)).to(dtype=x.dtype)

        # 3) Dual-band gated mixing
        g = self.sigmoid(self.mix_gate(torch.cat([y_lf, y_hf], dim=1)))
        y = g * y_hf + (1.0 - g) * y_lf

        # 4) Gain injection to spatial trunk
        gain = self.sigmoid(self.gain_proj(y))
        out = s * (1.0 + gain)
        return out


class _FSDESkipConnection(nn.Module):
    """
    SkipConnection with optional skip-feature enhancer.
    The enhancement is applied only to the skip branch (x), not to the submodule input.
    """

    def __init__(self, submodule: nn.Module, dim: int = 1, mode: str = "cat", enhancer: nn.Module | None = None):
        super().__init__()
        self.submodule = submodule
        self.dim = int(dim)
        self.mode = str(mode)
        self.enhancer = enhancer
        self.last_enhanced: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.submodule(x)
        skip = self.enhancer(x) if self.enhancer is not None else x
        self.last_enhanced = skip
        if self.mode == "cat":
            return torch.cat([skip, y], dim=self.dim)
        if self.mode == "add":
            return torch.add(skip, y)
        if self.mode == "mul":
            return torch.mul(skip, y)
        raise NotImplementedError(f"Unsupported mode {self.mode}.")


class UNet3DFullDualBandFSDE34(nn.Module):
    """
    Original MONAI UNet3D + Full Dual-Band FSDE on skip-3/4 (1-based).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
        strides: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_res_units: int = 0,
        fsde_skip_indices: Sequence[int] = (3, 4),  # 1-based over UNet skip levels
        fsde_input_size: Sequence[int] = (96, 96, 96),  # training ROI size
        fsde_alpha: float = 0.2,
        fsde_learnable_alpha: bool = False,
        fsde_alpha_max: float = 0.3,
        fsde_mix_gate_channels: int = 1,
        fsde_gain_channels: int = 1,
        fsde_use_norm_act: bool = True,
    ):
        super().__init__()
        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

        if len(channels) < 5:
            raise ValueError(f"UNet channels must have at least 5 stages, got {channels}")
        if len(fsde_input_size) != 3:
            raise ValueError(f"fsde_input_size must be 3D, got {fsde_input_size}")

        fsde_skip_indices = tuple(int(i) for i in fsde_skip_indices)
        valid_indices = set(range(1, len(channels)))  # 1..4 for default 5-level UNet
        invalid = [i for i in fsde_skip_indices if i not in valid_indices]
        if invalid:
            raise ValueError(
                f"fsde_skip_indices out of range {sorted(valid_indices)}: got {invalid}"
            )
        self.fsde_skip_indices = fsde_skip_indices
        self._skip_wrappers: dict[int, _FSDESkipConnection] = {}

        # Build per-skip enhancer map.
        d0, h0, w0 = (int(v) for v in fsde_input_size)
        self._fsde_enhancers: dict[int, nn.Module] = {}
        for skip_idx in fsde_skip_indices:
            # MONAI UNet first conv uses stride=2, so skip-i has spatial size / 2^i.
            d = max(1, d0 // (2 ** skip_idx))
            h = max(1, h0 // (2 ** skip_idx))
            w = max(1, w0 // (2 ** skip_idx))
            fft_shape = (d, h, w // 2 + 1)
            ch = int(channels[skip_idx - 1])
            self._fsde_enhancers[skip_idx] = FullDualBandFSDEBlock3D(
                channels=ch,
                fft_shape=fft_shape,
                alpha=fsde_alpha,
                learnable_alpha=fsde_learnable_alpha,
                alpha_max=fsde_alpha_max,
                mix_gate_channels=fsde_mix_gate_channels,
                gain_channels=fsde_gain_channels,
                use_norm_act=fsde_use_norm_act,
            )

        self._inject_fsde_skip_connections()

    def _inject_fsde_skip_connections(self) -> None:
        """
        Replace SkipConnection modules recursively and attach FSDE enhancer
        to target skip indices (1-based, shallow->deep).
        """
        counter = 0

        def _recurse(parent: nn.Module) -> None:
            nonlocal counter
            for name, child in list(parent.named_children()):
                if isinstance(child, SkipConnection):
                    counter += 1
                    enhancer = self._fsde_enhancers.get(counter, None)
                    wrapped = _FSDESkipConnection(
                        submodule=child.submodule,
                        dim=child.dim,
                        mode=child.mode,
                        enhancer=enhancer,
                    )
                    setattr(parent, name, wrapped)
                    self._skip_wrappers[counter] = wrapped
                    _recurse(wrapped.submodule)
                else:
                    _recurse(child)

        _recurse(self.net)

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


class UNet3DFullDualBandFSDE34Boundary(nn.Module):
    """
    UNet3D + Full Dual-Band FSDE on selected skips + boundary head on skip4.

    Boundary head is attached to the enhanced skip feature (after FSDE), then
    upsampled to the segmentation logits resolution for boundary supervision.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
        strides: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_res_units: int = 0,
        fsde_skip_indices: Sequence[int] = (4,),
        fsde_input_size: Sequence[int] = (96, 96, 96),
        fsde_alpha: float = 0.1,
        fsde_learnable_alpha: bool = False,
        fsde_alpha_max: float = 0.3,
        fsde_mix_gate_channels: int = 1,
        fsde_gain_channels: int = 1,
        fsde_use_norm_act: bool = True,
        boundary_skip_index: int = 4,
        boundary_mid_channels: int = 32,
        boundary_out_channels: int = 1,
        boundary_head_eval: bool = True,
    ):
        super().__init__()
        self.core = UNet3DFullDualBandFSDE34(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            fsde_skip_indices=fsde_skip_indices,
            fsde_input_size=fsde_input_size,
            fsde_alpha=fsde_alpha,
            fsde_learnable_alpha=fsde_learnable_alpha,
            fsde_alpha_max=fsde_alpha_max,
            fsde_mix_gate_channels=fsde_mix_gate_channels,
            fsde_gain_channels=fsde_gain_channels,
            fsde_use_norm_act=fsde_use_norm_act,
        )

        self.boundary_skip_index = int(boundary_skip_index)
        self.boundary_head_eval = bool(boundary_head_eval)
        if self.boundary_skip_index not in self.core._skip_wrappers:
            raise ValueError(
                f"boundary_skip_index={self.boundary_skip_index} not found in UNet skips."
            )
        if self.boundary_skip_index not in set(int(x) for x in fsde_skip_indices):
            raise ValueError(
                f"boundary_skip_index={self.boundary_skip_index} must be included in fsde_skip_indices="
                f"{tuple(int(x) for x in fsde_skip_indices)}."
            )
        if int(boundary_out_channels) <= 0:
            raise ValueError(f"boundary_out_channels must be > 0, got {boundary_out_channels}")

        boundary_in_channels = int(channels[self.boundary_skip_index - 1])
        boundary_mid_channels = int(boundary_mid_channels)
        if boundary_mid_channels <= 0:
            boundary_mid_channels = max(16, boundary_in_channels // 2)

        self.boundary_head = nn.Sequential(
            nn.Conv3d(
                boundary_in_channels,
                boundary_mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(boundary_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                boundary_mid_channels,
                int(boundary_out_channels),
                kernel_size=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, image, **batch):
        logits = self.core.net(image)
        output = {"logits": logits}

        # Emit boundary branch in training and optionally in eval.
        if self.training or self.boundary_head_eval:
            wrapper = self.core._skip_wrappers.get(self.boundary_skip_index)
            skip_feature = None if wrapper is None else wrapper.last_enhanced
            if skip_feature is None:
                raise RuntimeError(
                    "Boundary head could not read skip feature. "
                    "This usually means skip wrappers were not executed as expected."
                )
            boundary_logits = self.boundary_head(skip_feature)
            if tuple(boundary_logits.shape[2:]) != tuple(logits.shape[2:]):
                boundary_logits = F.interpolate(
                    boundary_logits,
                    size=logits.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            output["boundary_logits"] = boundary_logits
        return output

    def __str__(self):
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        info = super().__str__()
        info += f"\nAll parameters: {all_parameters}"
        info += f"\nTrainable parameters: {trainable_parameters}"
        return info


def _resolve_gn_groups(num_channels: int, max_groups: int) -> int:
    groups = max(1, min(int(max_groups), int(num_channels)))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class DualRouteFSDEBlock3D(nn.Module):
    """
    Dual-route FSDE block with GroupNorm.

    Route A (spatial): Conv3x3 + GN + SiLU
    Route B (frequency): Conv1x1 -> rFFT modulation -> iFFT
    Fusion: concat(A, B) -> Conv1x1 + GN + SiLU + residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gn_groups: int = 8,
        alpha: float = 0.1,
        alpha_max: float = 0.3,
        freq_kernel_size: Sequence[int] = (8, 8, 8),
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.alpha_max = float(alpha_max)

        if len(freq_kernel_size) != 3:
            raise ValueError(
                f"freq_kernel_size must be 3D, got {tuple(freq_kernel_size)}"
            )
        fd, fh, fw = (max(1, int(v)) for v in freq_kernel_size)

        g_spatial = _resolve_gn_groups(self.out_channels, gn_groups)
        g_fuse = _resolve_gn_groups(self.out_channels, gn_groups)

        self.spatial = nn.Sequential(
            nn.Conv3d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(g_spatial, self.out_channels),
            nn.SiLU(inplace=True),
        )

        self.freq_proj = nn.Sequential(
            nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(g_spatial, self.out_channels),
            nn.SiLU(inplace=True),
        )
        self.weight_lf = nn.Parameter(torch.zeros(1, self.out_channels, fd, fh, fw))
        self.weight_hf = nn.Parameter(torch.zeros(1, self.out_channels, fd, fh, fw))
        self.mix_gate = nn.Conv3d(self.out_channels * 2, self.out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

        self.fuse = nn.Sequential(
            nn.Conv3d(self.out_channels * 2, self.out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(g_fuse, self.out_channels),
            nn.SiLU(inplace=True),
        )

        if self.in_channels == self.out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(_resolve_gn_groups(self.out_channels, gn_groups), self.out_channels),
            )

    @staticmethod
    def _resize_freq_weight(weight: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if tuple(weight.shape[-3:]) == tuple(target_shape):
            return weight
        return F.interpolate(weight, size=target_shape, mode="trilinear", align_corners=False)

    def _alpha_value(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        value = self.alpha.to(dtype=dtype, device=device)
        return torch.clamp(value, 0.0, self.alpha_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(x)

        freq_feat = self.freq_proj(x).float()
        spectrum = torch.fft.rfftn(freq_feat, dim=(2, 3, 4))
        _, _, d_f, h_f, w_f = spectrum.shape
        target_shape = (d_f, h_f, w_f)

        w_lf = self._resize_freq_weight(self.weight_lf, target_shape)
        w_hf = self._resize_freq_weight(self.weight_hf, target_shape)
        alpha = self._alpha_value(dtype=w_lf.dtype, device=w_lf.device)
        m_lf = 1.0 + alpha * torch.tanh(w_lf)
        m_hf = 1.0 + alpha * torch.tanh(w_hf)

        y_lf = torch.fft.irfftn(spectrum * m_lf, s=freq_feat.shape[2:], dim=(2, 3, 4))
        y_hf = torch.fft.irfftn(spectrum * m_hf, s=freq_feat.shape[2:], dim=(2, 3, 4))
        y_lf = y_lf.to(dtype=x.dtype)
        y_hf = y_hf.to(dtype=x.dtype)

        gate = self.sigmoid(self.mix_gate(torch.cat([y_lf, y_hf], dim=1)))
        frequency = gate * y_hf + (1.0 - gate) * y_lf

        fused = self.fuse(torch.cat([spatial, frequency], dim=1))
        out = fused + self.residual(x)
        return out


class UNet3DDualRouteFSDEGN(nn.Module):
    """
    3D UNet variant where all encoder/decoder stages use dual-route FSDE blocks.
    - Norm: GroupNorm
    - Skip fusion: direct concatenation
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
        gn_groups: int = 8,
        fsde_alpha: float = 0.1,
        fsde_alpha_max: float = 0.3,
        freq_kernel_size: Sequence[int] = (8, 8, 8),
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(f"channels must contain 5 values, got {tuple(channels)}")
        c1, c2, c3, c4, c5 = (int(v) for v in channels)

        self.enc1 = DualRouteFSDEBlock3D(
            in_channels=in_channels,
            out_channels=c1,
            gn_groups=gn_groups,
            alpha=fsde_alpha,
            alpha_max=fsde_alpha_max,
            freq_kernel_size=freq_kernel_size,
        )
        self.down1 = nn.Conv3d(c1, c2, kernel_size=2, stride=2, bias=False)
        self.enc2 = DualRouteFSDEBlock3D(c2, c2, gn_groups, fsde_alpha, fsde_alpha_max, freq_kernel_size)
        self.down2 = nn.Conv3d(c2, c3, kernel_size=2, stride=2, bias=False)
        self.enc3 = DualRouteFSDEBlock3D(c3, c3, gn_groups, fsde_alpha, fsde_alpha_max, freq_kernel_size)
        self.down3 = nn.Conv3d(c3, c4, kernel_size=2, stride=2, bias=False)
        self.enc4 = DualRouteFSDEBlock3D(c4, c4, gn_groups, fsde_alpha, fsde_alpha_max, freq_kernel_size)
        self.down4 = nn.Conv3d(c4, c5, kernel_size=2, stride=2, bias=False)

        self.bottleneck = DualRouteFSDEBlock3D(c5, c5, gn_groups, fsde_alpha, fsde_alpha_max, freq_kernel_size)

        self.up4 = nn.ConvTranspose3d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DualRouteFSDEBlock3D(c4 + c4, c4, gn_groups, fsde_alpha, fsde_alpha_max, freq_kernel_size)
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DualRouteFSDEBlock3D(c3 + c3, c3, gn_groups, fsde_alpha, fsde_alpha_max, freq_kernel_size)
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DualRouteFSDEBlock3D(c2 + c2, c2, gn_groups, fsde_alpha, fsde_alpha_max, freq_kernel_size)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DualRouteFSDEBlock3D(c1 + c1, c1, gn_groups, fsde_alpha, fsde_alpha_max, freq_kernel_size)

        self.head = nn.Conv3d(c1, out_channels, kernel_size=1, bias=True)

    @staticmethod
    def _concat_skip(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if tuple(upsampled.shape[-3:]) != tuple(skip.shape[-3:]):
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        return torch.cat([skip, upsampled], dim=1)

    def forward(self, image, **batch):
        e1 = self.enc1(image)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))

        d4 = self.dec4(self._concat_skip(self.up4(b), e4))
        d3 = self.dec3(self._concat_skip(self.up3(d4), e3))
        d2 = self.dec2(self._concat_skip(self.up2(d3), e2))
        d1 = self.dec1(self._concat_skip(self.up1(d2), e1))
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
