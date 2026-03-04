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

        # Boundary supervision is needed only during training.
        if self.training:
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
