from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


def _resolve_device(device: str) -> torch.device:
    if str(device).lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _load_checkpoint(path: str) -> Mapping[str, torch.Tensor]:
    if not path:
        raise ValueError("checkpoint path is empty")
    ckpt_path = Path(path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state = payload["state_dict"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise ValueError(f"unsupported checkpoint format: {ckpt_path}")

    if any(key.startswith("module.") for key in state.keys()):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def _resize_scalar_nearest(label: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    value = label.float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(value, size=tuple(int(x) for x in size), mode="nearest")
    return resized[0, 0].long()


def _resize_volume_trilinear(volume: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    value = volume.float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        value,
        size=tuple(int(x) for x in size),
        mode="trilinear",
        align_corners=False,
    )
    return resized[0, 0]


def scalar_to_brats3ch(scalar: torch.Tensor) -> torch.Tensor:
    scalar = scalar.long()
    tc = ((scalar == 1) | (scalar == 3)).float()
    wt = ((scalar == 1) | (scalar == 2) | (scalar == 3)).float()
    et = (scalar == 3).float()
    return torch.stack([tc, wt, et], dim=0)


def _norm_minus1_1(scan: torch.Tensor):
    lo = torch.quantile(scan, 0.01)
    hi = torch.quantile(scan, 0.99)
    if not torch.isfinite(lo) or not torch.isfinite(hi) or float((hi - lo).abs().item()) < 1e-6:
        lo = scan.min()
        hi = scan.max()
    if float((hi - lo).abs().item()) < 1e-6:
        hi = lo + 1.0
    scaled = (scan - lo) / (hi - lo)
    scaled = torch.clamp(scaled, 0.0, 1.0)
    return scaled * 2.0 - 1.0, lo, hi


def _denorm_minus1_1(scan: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    scan = torch.clamp(scan, -1.0, 1.0)
    scaled = (scan + 1.0) * 0.5
    return scaled * (hi - lo) + lo


class LabelGANGenerator(nn.Module):
    """
    LabelGAN generator used by GliGAN repository for sampling synthetic label priors.
    """

    def __init__(self, noise: int = 100, channel: int = 64, out_channels: int = 3):
        super().__init__()
        c = int(channel)
        self.noise = int(noise)
        self.leaky_relu = nn.LeakyReLU()
        self.tp_conv1 = nn.ConvTranspose3d(self.noise, c * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm3d(c * 8)
        self.tp_conv2 = nn.Conv3d(c * 8, c * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(c * 4)
        self.tp_conv3 = nn.Conv3d(c * 4, c * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.InstanceNorm3d(c * 2)
        self.tp_conv4 = nn.Conv3d(c * 2, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.InstanceNorm3d(c)
        self.tp_conv5 = nn.Conv3d(c, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        noise = noise.view(-1, self.noise, 1, 1, 1)
        h = self.leaky_relu(self.bn1(self.tp_conv1(noise)))
        h = F.interpolate(h, scale_factor=2.0, mode="trilinear", align_corners=False)
        h = self.leaky_relu(self.bn2(self.tp_conv2(h)))
        h = F.interpolate(h, scale_factor=2.0, mode="trilinear", align_corners=False)
        h = self.leaky_relu(self.bn3(self.tp_conv3(h)))
        h = F.interpolate(h, scale_factor=2.0, mode="trilinear", align_corners=False)
        h = self.leaky_relu(self.bn4(self.tp_conv4(h)))
        h = F.interpolate(h, scale_factor=2.0, mode="trilinear", align_corners=False)
        h = self.tp_conv5(h)
        return torch.tanh(h)


class GliGANLabelSampler:
    def __init__(self, checkpoint_path: str, device: str, latent_dim: int = 100):
        self.device = _resolve_device(device)
        self.latent_dim = int(latent_dim)
        model = LabelGANGenerator(noise=self.latent_dim, out_channels=3)
        state = _load_checkpoint(checkpoint_path)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def sample_scalar_label(self, spatial_size: Sequence[int]) -> torch.Tensor:
        z = torch.randn((1, self.latent_dim), device=self.device)
        pred = self.model(z)[0]  # [3, d, h, w], tanh output
        pred = F.interpolate(
            pred.unsqueeze(0),
            size=tuple(int(x) for x in spatial_size),
            mode="trilinear",
            align_corners=False,
        )[0]
        binary = (pred > 0.5).long()
        summed = binary.sum(dim=0)
        scalar = torch.zeros_like(summed, dtype=torch.long)
        # Keep class mapping consistent with official GliGAN inference script.
        scalar[summed == 1] = 2
        scalar[summed == 2] = 1
        scalar[summed == 3] = 3
        return scalar


@dataclass
class GliGANTumourConfig:
    ckpt_t2f: str
    ckpt_t1c: str
    ckpt_t1n: str
    ckpt_t2w: str
    feature_size: int = 48
    use_checkpoint: bool = False
    img_size: tuple[int, int, int] = (96, 96, 96)
    device: str = "cpu"


class GliGANTumourSynthesizer:
    """
    Real GliGAN tumour synthesizer for 4 BraTS modalities:
    channel order must be [t2f, t1c, t1n, t2w].
    """

    def __init__(self, cfg: GliGANTumourConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.img_size = tuple(int(x) for x in cfg.img_size)
        ckpts = {
            "t2f": cfg.ckpt_t2f,
            "t1c": cfg.ckpt_t1c,
            "t1n": cfg.ckpt_t1n,
            "t2w": cfg.ckpt_t2w,
        }
        self.generators = nn.ModuleDict()
        for key, ckpt in ckpts.items():
            net = SwinUNETR(
                img_size=self.img_size,
                in_channels=4,
                out_channels=1,
                feature_size=int(cfg.feature_size),
                use_checkpoint=bool(cfg.use_checkpoint),
            )
            net.load_state_dict(_load_checkpoint(ckpt), strict=True)
            self.generators[key] = net
        self.generators.to(self.device)
        self.generators.eval()
        self._channel_order = ["t2f", "t1c", "t1n", "t2w"]

    @torch.no_grad()
    def synthesize(self, image: torch.Tensor, lesion_scalar: torch.Tensor, noise_std: float = 1.0) -> torch.Tensor:
        if image.ndim != 4 or image.shape[0] != 4:
            raise ValueError(f"GliGAN expects image [4,D,H,W], got {tuple(image.shape)}")
        if lesion_scalar.ndim != 3:
            raise ValueError(f"GliGAN expects lesion scalar [D,H,W], got {tuple(lesion_scalar.shape)}")

        src_size = tuple(int(x) for x in image.shape[-3:])
        if src_size != self.img_size:
            image_run = torch.stack(
                [_resize_volume_trilinear(image[i], self.img_size) for i in range(4)],
                dim=0,
            )
            lesion_run = _resize_scalar_nearest(lesion_scalar, self.img_size)
        else:
            image_run = image
            lesion_run = lesion_scalar

        lesion_mask = lesion_run > 0
        label_3ch = scalar_to_brats3ch(lesion_run).to(self.device)
        image_run = image_run.to(self.device).float()

        outputs = []
        for channel_idx, channel_name in enumerate(self._channel_order):
            healthy = image_run[channel_idx]
            healthy_norm, lo, hi = _norm_minus1_1(healthy)
            noisy = healthy_norm.clone()
            noise = torch.randn_like(noisy) * float(noise_std)
            noisy = torch.where(lesion_mask.to(self.device), noise, noisy)
            net_input = torch.cat([noisy.unsqueeze(0), label_3ch], dim=0).unsqueeze(0).float()
            pred_norm = self.generators[channel_name](net_input)[0, 0]
            pred = _denorm_minus1_1(pred_norm, lo=lo, hi=hi)
            outputs.append(pred)

        out = torch.stack(outputs, dim=0)
        if src_size != self.img_size:
            out = torch.stack([_resize_volume_trilinear(out[i], src_size) for i in range(4)], dim=0)
        return out.cpu()
