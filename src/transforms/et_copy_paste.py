from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from monai.transforms import MapTransform
from torch import nn


def _torch_load_compat(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class RandETCopyPasted(MapTransform):
    """
    On-the-fly ET-focused copy-paste augmentation for BraTS patches.

    Donor bank payload format:
    {
      "donors": [
        {
          "image": Tensor [4, D, H, W],
          "mask_et": Tensor [D, H, W] or [1, D, H, W],
          "et_voxels": int,
          "case_id": str,
        },
        ...
      ]
    }
    """

    def __init__(
        self,
        keys,
        donor_bank_path: str,
        prob: float = 0.4,
        max_insertions: int = 2,
        second_insertion_prob: float = 0.35,
        scale_range: Sequence[float] = (0.6, 1.0),
        flip_prob: float = 0.5,
        rot90_prob: float = 0.5,
        small_donor_bias: float = 1.0,
        min_insert_voxels: int = 24,
        min_distance_to_tumor: int = 6,
        blend_band_width: int = 3,
        blend_smooth_steps: int = 1,
        intensity_match: bool = True,
        intensity_eps: float = 1e-6,
        max_position_trials: int = 16,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if not donor_bank_path:
            raise ValueError("donor_bank_path must be provided.")
        if not (0.0 <= float(prob) <= 1.0):
            raise ValueError(f"prob must be in [0,1], got {prob}")
        if int(max_insertions) < 1:
            raise ValueError(f"max_insertions must be >= 1, got {max_insertions}")
        if not (0.0 <= float(second_insertion_prob) <= 1.0):
            raise ValueError(
                f"second_insertion_prob must be in [0,1], got {second_insertion_prob}"
            )
        if len(scale_range) != 2:
            raise ValueError("scale_range must have 2 values: [low, high]")
        scale_low = float(scale_range[0])
        scale_high = float(scale_range[1])
        if scale_low <= 0 or scale_high <= 0 or scale_low > scale_high:
            raise ValueError(f"invalid scale_range={scale_range}, require 0<low<=high")
        if not (0.0 <= float(flip_prob) <= 1.0):
            raise ValueError(f"flip_prob must be in [0,1], got {flip_prob}")
        if not (0.0 <= float(rot90_prob) <= 1.0):
            raise ValueError(f"rot90_prob must be in [0,1], got {rot90_prob}")
        if float(small_donor_bias) < 0:
            raise ValueError(f"small_donor_bias must be >=0, got {small_donor_bias}")
        if int(min_insert_voxels) < 1:
            raise ValueError(f"min_insert_voxels must be >=1, got {min_insert_voxels}")
        if int(min_distance_to_tumor) < 0:
            raise ValueError(
                f"min_distance_to_tumor must be >=0, got {min_distance_to_tumor}"
            )
        if int(blend_band_width) < 0:
            raise ValueError(f"blend_band_width must be >=0, got {blend_band_width}")
        if int(blend_smooth_steps) < 0:
            raise ValueError(f"blend_smooth_steps must be >=0, got {blend_smooth_steps}")
        if float(intensity_eps) <= 0:
            raise ValueError(f"intensity_eps must be >0, got {intensity_eps}")
        if int(max_position_trials) < 1:
            raise ValueError(f"max_position_trials must be >=1, got {max_position_trials}")

        self.donor_bank_path = str(donor_bank_path)
        self.prob = float(prob)
        self.max_insertions = int(max_insertions)
        self.second_insertion_prob = float(second_insertion_prob)
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.flip_prob = float(flip_prob)
        self.rot90_prob = float(rot90_prob)
        self.small_donor_bias = float(small_donor_bias)
        self.min_insert_voxels = int(min_insert_voxels)
        self.min_distance_to_tumor = int(min_distance_to_tumor)
        self.blend_band_width = int(blend_band_width)
        self.blend_smooth_steps = int(blend_smooth_steps)
        self.intensity_match = bool(intensity_match)
        self.intensity_eps = float(intensity_eps)
        self.max_position_trials = int(max_position_trials)

        self._donors = None
        self._donor_weights = None

    @staticmethod
    def _to_scalar_label(label: torch.Tensor):
        if label.ndim == 3:
            return label.long(), "scalar_no_channel"
        if label.ndim == 4 and label.shape[0] == 1:
            return label[0].long(), "scalar_with_channel"
        if label.ndim == 4 and label.shape[0] == 3:
            tc = label[0] > 0.5
            wt = label[1] > 0.5
            et = label[2] > 0.5
            scalar = torch.zeros_like(label[0], dtype=torch.long)
            scalar[wt] = 2
            scalar[tc] = 1
            scalar[et] = 3
            return scalar, "three_channel"
        raise ValueError(
            f"Unsupported label shape for ET copy-paste: {tuple(label.shape)}"
        )

    @staticmethod
    def _from_scalar_label(scalar: torch.Tensor, mode: str, dtype: torch.dtype):
        if mode == "scalar_no_channel":
            return scalar.to(dtype=dtype)
        if mode == "scalar_with_channel":
            return scalar.unsqueeze(0).to(dtype=dtype)
        if mode == "three_channel":
            tc = ((scalar == 1) | (scalar == 3)).to(dtype=dtype)
            wt = ((scalar == 1) | (scalar == 2) | (scalar == 3)).to(dtype=dtype)
            et = (scalar == 3).to(dtype=dtype)
            return torch.stack([tc, wt, et], dim=0)
        raise ValueError(f"Unknown label mode: {mode}")

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        value = mask.float().unsqueeze(0).unsqueeze(0)
        kernel = int(2 * radius + 1)
        dilated = F.max_pool3d(value, kernel_size=kernel, stride=1, padding=radius)
        return dilated[0, 0] > 0

    @staticmethod
    def _build_small_donor_weights(volumes: list[int], bias: float) -> torch.Tensor:
        vol = torch.as_tensor(volumes, dtype=torch.float32).clamp_min(1.0)
        if bias <= 0:
            weights = torch.ones_like(vol)
        else:
            weights = 1.0 / torch.pow(vol, bias)
        weights = weights / weights.sum().clamp_min(1e-8)
        return weights

    def _ensure_donor_bank(self) -> None:
        if self._donors is not None:
            return

        path = Path(self.donor_bank_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ET donor bank not found: {path}")

        payload = _torch_load_compat(path)
        if isinstance(payload, dict) and isinstance(payload.get("donors"), list):
            raw_donors = payload["donors"]
        elif isinstance(payload, list):
            raw_donors = payload
        else:
            raise ValueError(
                "Unsupported donor bank format. Expect dict with key 'donors' or a list."
            )

        donors = []
        volumes = []
        for item in raw_donors:
            if not isinstance(item, dict):
                continue
            image = torch.as_tensor(item.get("image", None))
            mask = item.get("mask_et", item.get("mask", None))
            if mask is None:
                continue
            mask = torch.as_tensor(mask)
            if image.ndim != 4 or image.shape[0] != 4:
                continue
            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 3 or tuple(mask.shape) != tuple(image.shape[-3:]):
                continue
            mask = mask > 0
            et_voxels = int(mask.sum().item())
            if et_voxels < self.min_insert_voxels:
                continue

            donors.append(
                {
                    "image": image.float().cpu().contiguous(),
                    "mask_et": mask.cpu().contiguous(),
                    "mask_wt": (torch.as_tensor(item.get("mask_wt", mask)) > 0)
                    .cpu()
                    .contiguous(),
                    "mask_tc": (torch.as_tensor(item.get("mask_tc", mask)) > 0)
                    .cpu()
                    .contiguous(),
                    "et_voxels": et_voxels,
                    "case_id": str(item.get("case_id", "")),
                }
            )
            volumes.append(et_voxels)

        if len(donors) == 0:
            raise ValueError(
                f"No valid donors found in ET donor bank: {path}. "
                f"Need donors with image [4,D,H,W] and mask_et [D,H,W]."
            )

        self._donors = donors
        self._donor_weights = self._build_small_donor_weights(
            volumes=volumes,
            bias=self.small_donor_bias,
        )

    def _sample_num_insertions(self) -> int:
        n = 1
        while n < self.max_insertions:
            if float(torch.rand(1).item()) < self.second_insertion_prob:
                n += 1
            else:
                break
        return n

    def _sample_donor(self):
        idx = int(torch.multinomial(self._donor_weights, num_samples=1).item())
        donor = self._donors[idx]
        return (
            donor["image"].clone(),
            {
                "et": donor["mask_et"].clone(),
                "wt": donor["mask_wt"].clone(),
                "tc": donor["mask_tc"].clone(),
            },
        )

    def _augment_donor(
        self, donor_image: torch.Tensor, donor_masks: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        donor_et = donor_masks["et"]
        donor_wt = donor_masks["wt"]
        donor_tc = donor_masks["tc"]
        scale = float(torch.empty(1).uniform_(self.scale_low, self.scale_high).item())
        if abs(scale - 1.0) > 1e-3:
            d, h, w = (int(x) for x in donor_et.shape)
            out_size = (
                max(1, int(round(d * scale))),
                max(1, int(round(h * scale))),
                max(1, int(round(w * scale))),
            )
            donor_image = F.interpolate(
                donor_image.unsqueeze(0),
                size=out_size,
                mode="trilinear",
                align_corners=False,
            )[0]
            donor_et = (
                F.interpolate(
                    donor_et.float().unsqueeze(0).unsqueeze(0),
                    size=out_size,
                    mode="nearest",
                )[0, 0]
                > 0.5
            )
            donor_wt = (
                F.interpolate(
                    donor_wt.float().unsqueeze(0).unsqueeze(0),
                    size=out_size,
                    mode="nearest",
                )[0, 0]
                > 0.5
            )
            donor_tc = (
                F.interpolate(
                    donor_tc.float().unsqueeze(0).unsqueeze(0),
                    size=out_size,
                    mode="nearest",
                )[0, 0]
                > 0.5
            )

        for axis in range(3):
            if float(torch.rand(1).item()) < self.flip_prob:
                donor_image = torch.flip(donor_image, dims=[axis + 1])
                donor_et = torch.flip(donor_et, dims=[axis])
                donor_wt = torch.flip(donor_wt, dims=[axis])
                donor_tc = torch.flip(donor_tc, dims=[axis])

        if float(torch.rand(1).item()) < self.rot90_prob:
            plane = [(0, 1), (0, 2), (1, 2)][int(torch.randint(0, 3, (1,)).item())]
            k = int(torch.randint(1, 4, (1,)).item())
            donor_image = torch.rot90(donor_image, k=k, dims=[plane[0] + 1, plane[1] + 1])
            donor_et = torch.rot90(donor_et, k=k, dims=list(plane))
            donor_wt = torch.rot90(donor_wt, k=k, dims=list(plane))
            donor_tc = torch.rot90(donor_tc, k=k, dims=list(plane))

        # Enforce hierarchy: ET subset TC subset WT.
        donor_et = donor_et > 0
        donor_tc = (donor_tc > 0) | donor_et
        donor_wt = (donor_wt > 0) | donor_tc
        return donor_image, {"et": donor_et, "wt": donor_wt, "tc": donor_tc}

    @staticmethod
    def _compute_overlap(
        center: tuple[int, int, int],
        target_shape: tuple[int, int, int],
        source_shape: tuple[int, int, int],
    ):
        d, h, w = target_shape
        sd, sh, sw = source_shape
        cz, cy, cx = center

        z0_raw = cz - sd // 2
        y0_raw = cy - sh // 2
        x0_raw = cx - sw // 2
        z1_raw = z0_raw + sd
        y1_raw = y0_raw + sh
        x1_raw = x0_raw + sw

        z0 = max(0, z0_raw)
        y0 = max(0, y0_raw)
        x0 = max(0, x0_raw)
        z1 = min(d, z1_raw)
        y1 = min(h, y1_raw)
        x1 = min(w, x1_raw)
        if min(z1 - z0, y1 - y0, x1 - x0) <= 0:
            return None

        src_z0 = max(0, -z0_raw)
        src_y0 = max(0, -y0_raw)
        src_x0 = max(0, -x0_raw)
        src_z1 = src_z0 + (z1 - z0)
        src_y1 = src_y0 + (y1 - y0)
        src_x1 = src_x0 + (x1 - x0)
        return (z0, z1, y0, y1, x0, x1, src_z0, src_z1, src_y0, src_y1, src_x0, src_x1)

    def _find_valid_placement(
        self,
        scalar_label: torch.Tensor,
        donor_masks: dict[str, torch.Tensor],
        dilated_tumor: torch.Tensor,
    ):
        d, h, w = (int(x) for x in scalar_label.shape)
        donor_wt = donor_masks["wt"]
        donor_et = donor_masks["et"]
        donor_tc = donor_masks["tc"]
        source_shape = tuple(int(x) for x in donor_wt.shape)
        target_shape = (d, h, w)

        for _ in range(self.max_position_trials):
            center = (
                int(torch.randint(0, d, (1,)).item()),
                int(torch.randint(0, h, (1,)).item()),
                int(torch.randint(0, w, (1,)).item()),
            )
            overlap = self._compute_overlap(
                center=center,
                target_shape=target_shape,
                source_shape=source_shape,
            )
            if overlap is None:
                continue

            (
                z0,
                z1,
                y0,
                y1,
                x0,
                x1,
                src_z0,
                src_z1,
                src_y0,
                src_y1,
                src_x0,
                src_x1,
            ) = overlap

            target_region = scalar_label[z0:z1, y0:y1, x0:x1]
            source_wt = donor_wt[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
            source_et = donor_et[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
            source_tc = donor_tc[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]

            valid_wt = source_wt & (target_region == 0)
            if self.min_distance_to_tumor > 0:
                valid_wt = valid_wt & (~dilated_tumor[z0:z1, y0:y1, x0:x1])
            valid_et = source_et & valid_wt
            valid_tc = source_tc & valid_wt

            if int(valid_et.sum().item()) >= self.min_insert_voxels:
                return {
                    "z0": z0,
                    "z1": z1,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                    "src_z0": src_z0,
                    "src_z1": src_z1,
                    "src_y0": src_y0,
                    "src_y1": src_y1,
                    "src_x0": src_x0,
                    "src_x1": src_x1,
                    "valid_wt": valid_wt,
                    "valid_tc": valid_tc,
                    "valid_et": valid_et,
                }

        return None

    def _match_intensity(
        self,
        donor_region: torch.Tensor,
        target_region: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        matched = donor_region.clone()
        voxel_count = int(valid_mask.sum().item())
        if voxel_count < 8:
            return matched

        for c in range(matched.shape[0]):
            src_vals = matched[c][valid_mask]
            tgt_vals = target_region[c][valid_mask]
            if int(src_vals.numel()) < 8 or int(tgt_vals.numel()) < 8:
                continue
            src_mean = src_vals.mean()
            src_std = src_vals.std(unbiased=False)
            tgt_mean = tgt_vals.mean()
            tgt_std = tgt_vals.std(unbiased=False)
            matched[c] = (matched[c] - src_mean) / (src_std + self.intensity_eps)
            matched[c] = matched[c] * (tgt_std + self.intensity_eps) + tgt_mean
        return matched

    def _build_soft_mask(self, hard_mask: torch.Tensor) -> torch.Tensor:
        soft = hard_mask.float()
        if self.blend_band_width > 0:
            soft = F.max_pool3d(
                soft.unsqueeze(0).unsqueeze(0),
                kernel_size=int(2 * self.blend_band_width + 1),
                stride=1,
                padding=int(self.blend_band_width),
            )[0, 0]
        for _ in range(self.blend_smooth_steps):
            soft = F.avg_pool3d(
                soft.unsqueeze(0).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1,
            )[0, 0]
        soft = torch.clamp(soft, 0.0, 1.0)
        soft = soft * (soft > 0).float()
        soft[hard_mask] = 1.0
        return soft

    def __call__(self, data):
        output = dict(data)
        if float(torch.rand(1).item()) >= self.prob:
            return output

        self._ensure_donor_bank()

        image = torch.as_tensor(output["image"]).clone().float()
        label = torch.as_tensor(output["label"]).clone()
        if image.ndim != 4 or image.shape[0] != 4:
            raise ValueError(
                f"RandETCopyPasted expects image [4,D,H,W], got {tuple(image.shape)}"
            )

        label_dtype = label.dtype
        scalar_label, label_mode = self._to_scalar_label(label)
        tumor_mask = scalar_label > 0
        dilated_tumor = self._dilate_mask(
            tumor_mask,
            radius=self.min_distance_to_tumor,
        )

        n_insert = self._sample_num_insertions()
        for _ in range(n_insert):
            donor_image, donor_masks = self._sample_donor()
            donor_image, donor_masks = self._augment_donor(donor_image, donor_masks)
            placement = self._find_valid_placement(
                scalar_label=scalar_label,
                donor_masks=donor_masks,
                dilated_tumor=dilated_tumor,
            )
            if placement is None:
                continue

            z0, z1 = placement["z0"], placement["z1"]
            y0, y1 = placement["y0"], placement["y1"]
            x0, x1 = placement["x0"], placement["x1"]
            src_z0, src_z1 = placement["src_z0"], placement["src_z1"]
            src_y0, src_y1 = placement["src_y0"], placement["src_y1"]
            src_x0, src_x1 = placement["src_x0"], placement["src_x1"]
            valid_wt = placement["valid_wt"]
            valid_tc = placement["valid_tc"]
            valid_et = placement["valid_et"]

            target_region = image[:, z0:z1, y0:y1, x0:x1]
            donor_region = donor_image[
                :,
                src_z0:src_z1,
                src_y0:src_y1,
                src_x0:src_x1,
            ]
            if self.intensity_match:
                donor_region = self._match_intensity(
                    donor_region=donor_region,
                    target_region=target_region,
                    valid_mask=valid_wt,
                )

            soft_mask = self._build_soft_mask(valid_wt).unsqueeze(0)
            image[:, z0:z1, y0:y1, x0:x1] = (
                target_region * (1.0 - soft_mask) + donor_region * soft_mask
            )
            scalar_label_region = scalar_label[z0:z1, y0:y1, x0:x1].clone()
            # Keep hierarchy WT ⊇ TC ⊇ ET in scalar space:
            # WT-only -> 2, TC -> 1, ET -> 3 (ET overrides TC/WT).
            scalar_label_region[valid_wt] = 2
            scalar_label_region[valid_tc] = 1
            scalar_label_region[valid_et] = 3
            scalar_label[z0:z1, y0:y1, x0:x1] = scalar_label_region

            tumor_mask = scalar_label > 0
            dilated_tumor = self._dilate_mask(
                tumor_mask,
                radius=self.min_distance_to_tumor,
            )

        output["image"] = image
        output["label"] = self._from_scalar_label(
            scalar=scalar_label,
            mode=label_mode,
            dtype=label_dtype,
        )
        return output


class RandETCopyPasteBatch(nn.Module):
    """
    GPU-oriented ET copy-paste augmentation operating on full training batch.

    Expected batch keys:
    - image: [B, 4, D, H, W]
    - label: [B, C, D, H, W], C can be 1 or 3
    """

    def __init__(
        self,
        donor_bank_path: str,
        image_key: str = "image",
        label_key: str = "label",
        prob: float = 0.25,
        max_insertions: int = 1,
        second_insertion_prob: float = 0.0,
        scale_range: Sequence[float] = (0.8, 1.0),
        flip_prob: float = 0.3,
        rot90_prob: float = 0.3,
        small_donor_bias: float = 1.0,
        min_insert_voxels: int = 24,
        min_distance_to_tumor: int = 6,
        blend_band_width: int = 2,
        blend_smooth_steps: int = 0,
        intensity_match: bool = False,
        intensity_eps: float = 1e-6,
        max_position_trials: int = 8,
        cache_on_device: bool = True,
    ):
        super().__init__()
        if not donor_bank_path:
            raise ValueError("donor_bank_path must be provided.")
        if not (0.0 <= float(prob) <= 1.0):
            raise ValueError(f"prob must be in [0,1], got {prob}")
        if int(max_insertions) < 1:
            raise ValueError(f"max_insertions must be >= 1, got {max_insertions}")
        if not (0.0 <= float(second_insertion_prob) <= 1.0):
            raise ValueError(
                f"second_insertion_prob must be in [0,1], got {second_insertion_prob}"
            )
        if len(scale_range) != 2:
            raise ValueError("scale_range must have 2 values: [low, high]")
        scale_low = float(scale_range[0])
        scale_high = float(scale_range[1])
        if scale_low <= 0 or scale_high <= 0 or scale_low > scale_high:
            raise ValueError(f"invalid scale_range={scale_range}, require 0<low<=high")
        if not (0.0 <= float(flip_prob) <= 1.0):
            raise ValueError(f"flip_prob must be in [0,1], got {flip_prob}")
        if not (0.0 <= float(rot90_prob) <= 1.0):
            raise ValueError(f"rot90_prob must be in [0,1], got {rot90_prob}")
        if float(small_donor_bias) < 0:
            raise ValueError(f"small_donor_bias must be >=0, got {small_donor_bias}")
        if int(min_insert_voxels) < 1:
            raise ValueError(f"min_insert_voxels must be >=1, got {min_insert_voxels}")
        if int(min_distance_to_tumor) < 0:
            raise ValueError(
                f"min_distance_to_tumor must be >=0, got {min_distance_to_tumor}"
            )
        if int(blend_band_width) < 0:
            raise ValueError(f"blend_band_width must be >=0, got {blend_band_width}")
        if int(blend_smooth_steps) < 0:
            raise ValueError(f"blend_smooth_steps must be >=0, got {blend_smooth_steps}")
        if float(intensity_eps) <= 0:
            raise ValueError(f"intensity_eps must be >0, got {intensity_eps}")
        if int(max_position_trials) < 1:
            raise ValueError(f"max_position_trials must be >=1, got {max_position_trials}")

        self.donor_bank_path = str(donor_bank_path)
        self.image_key = str(image_key)
        self.label_key = str(label_key)
        self.prob = float(prob)
        self.max_insertions = int(max_insertions)
        self.second_insertion_prob = float(second_insertion_prob)
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.flip_prob = float(flip_prob)
        self.rot90_prob = float(rot90_prob)
        self.small_donor_bias = float(small_donor_bias)
        self.min_insert_voxels = int(min_insert_voxels)
        self.min_distance_to_tumor = int(min_distance_to_tumor)
        self.blend_band_width = int(blend_band_width)
        self.blend_smooth_steps = int(blend_smooth_steps)
        self.intensity_match = bool(intensity_match)
        self.intensity_eps = float(intensity_eps)
        self.max_position_trials = int(max_position_trials)
        self.cache_on_device = bool(cache_on_device)

        self._donors_cpu = None
        self._donor_weights_cpu = None
        self._donors_device = None
        self._donor_weights_device = None
        self._cached_device = None

    @staticmethod
    def _to_scalar_label(label: torch.Tensor):
        if label.ndim == 3:
            return label.long(), "scalar_no_channel"
        if label.ndim == 4 and label.shape[0] == 1:
            return label[0].long(), "scalar_with_channel"
        if label.ndim == 4 and label.shape[0] == 3:
            tc = label[0] > 0.5
            wt = label[1] > 0.5
            et = label[2] > 0.5
            scalar = torch.zeros_like(label[0], dtype=torch.long)
            scalar[wt] = 2
            scalar[tc] = 1
            scalar[et] = 3
            return scalar, "three_channel"
        raise ValueError(
            f"Unsupported label shape for ET copy-paste batch: {tuple(label.shape)}"
        )

    @staticmethod
    def _from_scalar_label(scalar: torch.Tensor, mode: str, dtype: torch.dtype):
        if mode == "scalar_no_channel":
            return scalar.to(dtype=dtype)
        if mode == "scalar_with_channel":
            return scalar.unsqueeze(0).to(dtype=dtype)
        if mode == "three_channel":
            tc = ((scalar == 1) | (scalar == 3)).to(dtype=dtype)
            wt = ((scalar == 1) | (scalar == 2) | (scalar == 3)).to(dtype=dtype)
            et = (scalar == 3).to(dtype=dtype)
            return torch.stack([tc, wt, et], dim=0)
        raise ValueError(f"Unknown label mode: {mode}")

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        value = mask.float().unsqueeze(0).unsqueeze(0)
        kernel = int(2 * radius + 1)
        dilated = F.max_pool3d(value, kernel_size=kernel, stride=1, padding=radius)
        return dilated[0, 0] > 0

    @staticmethod
    def _build_small_donor_weights(volumes: list[int], bias: float) -> torch.Tensor:
        vol = torch.as_tensor(volumes, dtype=torch.float32).clamp_min(1.0)
        if bias <= 0:
            weights = torch.ones_like(vol)
        else:
            weights = 1.0 / torch.pow(vol, bias)
        weights = weights / weights.sum().clamp_min(1e-8)
        return weights

    def _ensure_donor_bank_cpu(self) -> None:
        if self._donors_cpu is not None:
            return

        path = Path(self.donor_bank_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ET donor bank not found: {path}")
        payload = _torch_load_compat(path)
        if isinstance(payload, dict) and isinstance(payload.get("donors"), list):
            raw_donors = payload["donors"]
        elif isinstance(payload, list):
            raw_donors = payload
        else:
            raise ValueError(
                "Unsupported donor bank format. Expect dict with key 'donors' or a list."
            )

        donors = []
        volumes = []
        for item in raw_donors:
            if not isinstance(item, dict):
                continue
            image = torch.as_tensor(item.get("image", None))
            mask = item.get("mask_et", item.get("mask", None))
            if mask is None:
                continue
            mask = torch.as_tensor(mask)
            if image.ndim != 4 or image.shape[0] != 4:
                continue
            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 3 or tuple(mask.shape) != tuple(image.shape[-3:]):
                continue
            mask = mask > 0
            et_voxels = int(mask.sum().item())
            if et_voxels < self.min_insert_voxels:
                continue

            donors.append(
                {
                    "image": image.float().cpu().contiguous(),
                    "mask_et": mask.cpu().contiguous(),
                    "mask_wt": (torch.as_tensor(item.get("mask_wt", mask)) > 0)
                    .cpu()
                    .contiguous(),
                    "mask_tc": (torch.as_tensor(item.get("mask_tc", mask)) > 0)
                    .cpu()
                    .contiguous(),
                }
            )
            volumes.append(et_voxels)

        if len(donors) == 0:
            raise ValueError(
                f"No valid donors found in ET donor bank: {path}. "
                f"Need donors with image [4,D,H,W] and mask_et [D,H,W]."
            )

        self._donors_cpu = donors
        self._donor_weights_cpu = self._build_small_donor_weights(
            volumes=volumes,
            bias=self.small_donor_bias,
        )

    def _ensure_donor_bank_device(self, device: torch.device) -> None:
        self._ensure_donor_bank_cpu()
        if not self.cache_on_device:
            return
        if self._cached_device == device and self._donors_device is not None:
            return

        self._donors_device = [
            {
                "image": donor["image"].to(device, non_blocking=True),
                "mask_et": donor["mask_et"].to(device, non_blocking=True),
                "mask_wt": donor["mask_wt"].to(device, non_blocking=True),
                "mask_tc": donor["mask_tc"].to(device, non_blocking=True),
            }
            for donor in self._donors_cpu
        ]
        self._donor_weights_device = self._donor_weights_cpu.to(device, non_blocking=True)
        self._cached_device = device

    def _sample_num_insertions(self) -> int:
        n = 1
        while n < self.max_insertions:
            if random.random() < self.second_insertion_prob:
                n += 1
            else:
                break
        return n

    def _sample_donor(
        self, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.cache_on_device:
            idx = int(torch.multinomial(self._donor_weights_device, num_samples=1).item())
            donor = self._donors_device[idx]
            return (
                donor["image"].clone(),
                {
                    "et": donor["mask_et"].clone(),
                    "wt": donor["mask_wt"].clone(),
                    "tc": donor["mask_tc"].clone(),
                },
            )

        idx = int(torch.multinomial(self._donor_weights_cpu, num_samples=1).item())
        donor = self._donors_cpu[idx]
        return (
            donor["image"].to(device, non_blocking=True),
            {
                "et": donor["mask_et"].to(device, non_blocking=True),
                "wt": donor["mask_wt"].to(device, non_blocking=True),
                "tc": donor["mask_tc"].to(device, non_blocking=True),
            },
        )

    def _augment_donor(
        self, donor_image: torch.Tensor, donor_masks: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        donor_et = donor_masks["et"]
        donor_wt = donor_masks["wt"]
        donor_tc = donor_masks["tc"]
        scale = random.uniform(self.scale_low, self.scale_high)
        if abs(scale - 1.0) > 1e-3:
            d, h, w = (int(x) for x in donor_et.shape)
            out_size = (
                max(1, int(round(d * scale))),
                max(1, int(round(h * scale))),
                max(1, int(round(w * scale))),
            )
            donor_image = F.interpolate(
                donor_image.unsqueeze(0),
                size=out_size,
                mode="trilinear",
                align_corners=False,
            )[0]
            donor_et = (
                F.interpolate(
                    donor_et.float().unsqueeze(0).unsqueeze(0),
                    size=out_size,
                    mode="nearest",
                )[0, 0]
                > 0.5
            )
            donor_wt = (
                F.interpolate(
                    donor_wt.float().unsqueeze(0).unsqueeze(0),
                    size=out_size,
                    mode="nearest",
                )[0, 0]
                > 0.5
            )
            donor_tc = (
                F.interpolate(
                    donor_tc.float().unsqueeze(0).unsqueeze(0),
                    size=out_size,
                    mode="nearest",
                )[0, 0]
                > 0.5
            )

        for axis in range(3):
            if random.random() < self.flip_prob:
                donor_image = torch.flip(donor_image, dims=[axis + 1])
                donor_et = torch.flip(donor_et, dims=[axis])
                donor_wt = torch.flip(donor_wt, dims=[axis])
                donor_tc = torch.flip(donor_tc, dims=[axis])

        if random.random() < self.rot90_prob:
            plane = [(0, 1), (0, 2), (1, 2)][random.randrange(3)]
            k = random.randrange(1, 4)
            donor_image = torch.rot90(donor_image, k=k, dims=[plane[0] + 1, plane[1] + 1])
            donor_et = torch.rot90(donor_et, k=k, dims=list(plane))
            donor_wt = torch.rot90(donor_wt, k=k, dims=list(plane))
            donor_tc = torch.rot90(donor_tc, k=k, dims=list(plane))

        donor_et = donor_et > 0
        donor_tc = (donor_tc > 0) | donor_et
        donor_wt = (donor_wt > 0) | donor_tc
        return donor_image, {"et": donor_et, "wt": donor_wt, "tc": donor_tc}

    @staticmethod
    def _compute_overlap(
        center: tuple[int, int, int],
        target_shape: tuple[int, int, int],
        source_shape: tuple[int, int, int],
    ):
        d, h, w = target_shape
        sd, sh, sw = source_shape
        cz, cy, cx = center

        z0_raw = cz - sd // 2
        y0_raw = cy - sh // 2
        x0_raw = cx - sw // 2
        z1_raw = z0_raw + sd
        y1_raw = y0_raw + sh
        x1_raw = x0_raw + sw

        z0 = max(0, z0_raw)
        y0 = max(0, y0_raw)
        x0 = max(0, x0_raw)
        z1 = min(d, z1_raw)
        y1 = min(h, y1_raw)
        x1 = min(w, x1_raw)
        if min(z1 - z0, y1 - y0, x1 - x0) <= 0:
            return None

        src_z0 = max(0, -z0_raw)
        src_y0 = max(0, -y0_raw)
        src_x0 = max(0, -x0_raw)
        src_z1 = src_z0 + (z1 - z0)
        src_y1 = src_y0 + (y1 - y0)
        src_x1 = src_x0 + (x1 - x0)
        return (z0, z1, y0, y1, x0, x1, src_z0, src_z1, src_y0, src_y1, src_x0, src_x1)

    def _find_valid_placement(
        self,
        scalar_label: torch.Tensor,
        donor_masks: dict[str, torch.Tensor],
        dilated_tumor: torch.Tensor,
    ):
        d, h, w = (int(x) for x in scalar_label.shape)
        donor_wt = donor_masks["wt"]
        donor_et = donor_masks["et"]
        donor_tc = donor_masks["tc"]
        source_shape = tuple(int(x) for x in donor_wt.shape)
        target_shape = (d, h, w)

        for _ in range(self.max_position_trials):
            center = (
                random.randrange(0, d),
                random.randrange(0, h),
                random.randrange(0, w),
            )
            overlap = self._compute_overlap(
                center=center,
                target_shape=target_shape,
                source_shape=source_shape,
            )
            if overlap is None:
                continue

            (
                z0,
                z1,
                y0,
                y1,
                x0,
                x1,
                src_z0,
                src_z1,
                src_y0,
                src_y1,
                src_x0,
                src_x1,
            ) = overlap
            target_region = scalar_label[z0:z1, y0:y1, x0:x1]
            source_wt = donor_wt[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
            source_et = donor_et[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
            source_tc = donor_tc[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]

            valid_wt = source_wt & (target_region == 0)
            if self.min_distance_to_tumor > 0:
                valid_wt = valid_wt & (~dilated_tumor[z0:z1, y0:y1, x0:x1])
            valid_et = source_et & valid_wt
            valid_tc = source_tc & valid_wt
            if int(valid_et.sum().item()) >= self.min_insert_voxels:
                return {
                    "z0": z0,
                    "z1": z1,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                    "src_z0": src_z0,
                    "src_z1": src_z1,
                    "src_y0": src_y0,
                    "src_y1": src_y1,
                    "src_x0": src_x0,
                    "src_x1": src_x1,
                    "valid_wt": valid_wt,
                    "valid_tc": valid_tc,
                    "valid_et": valid_et,
                }
        return None

    def _match_intensity(
        self,
        donor_region: torch.Tensor,
        target_region: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        matched = donor_region.clone()
        voxel_count = int(valid_mask.sum().item())
        if voxel_count < 8:
            return matched

        for c in range(matched.shape[0]):
            src_vals = matched[c][valid_mask]
            tgt_vals = target_region[c][valid_mask]
            if int(src_vals.numel()) < 8 or int(tgt_vals.numel()) < 8:
                continue
            src_mean = src_vals.mean()
            src_std = src_vals.std(unbiased=False)
            tgt_mean = tgt_vals.mean()
            tgt_std = tgt_vals.std(unbiased=False)
            matched[c] = (matched[c] - src_mean) / (src_std + self.intensity_eps)
            matched[c] = matched[c] * (tgt_std + self.intensity_eps) + tgt_mean
        return matched

    def _build_soft_mask(self, hard_mask: torch.Tensor) -> torch.Tensor:
        soft = hard_mask.float()
        if self.blend_band_width > 0:
            soft = F.max_pool3d(
                soft.unsqueeze(0).unsqueeze(0),
                kernel_size=int(2 * self.blend_band_width + 1),
                stride=1,
                padding=int(self.blend_band_width),
            )[0, 0]
        for _ in range(self.blend_smooth_steps):
            soft = F.avg_pool3d(
                soft.unsqueeze(0).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1,
            )[0, 0]
        soft = torch.clamp(soft, 0.0, 1.0)
        soft = soft * (soft > 0).float()
        soft[hard_mask] = 1.0
        return soft

    @torch.no_grad()
    def forward(self, batch: dict) -> dict:
        if not isinstance(batch, dict):
            raise ValueError(
                f"RandETCopyPasteBatch expects dict batch, got {type(batch)!r}"
            )
        if self.image_key not in batch or self.label_key not in batch:
            return batch

        image = batch[self.image_key]
        label = batch[self.label_key]
        if not torch.is_tensor(image) or not torch.is_tensor(label):
            return batch
        if image.ndim != 5 or image.shape[1] != 4:
            raise ValueError(
                f"Expected image [B,4,D,H,W], got {tuple(image.shape)}"
            )
        if label.ndim != 5:
            raise ValueError(
                f"Expected label [B,C,D,H,W], got {tuple(label.shape)}"
            )

        device = image.device
        self._ensure_donor_bank_device(device=device)

        image = image.clone().float()
        label = label.clone()

        for bidx in range(image.shape[0]):
            if random.random() >= self.prob:
                continue

            sample_image = image[bidx]
            sample_label = label[bidx]
            label_dtype = sample_label.dtype
            scalar_label, label_mode = self._to_scalar_label(sample_label)

            tumor_mask = scalar_label > 0
            dilated_tumor = self._dilate_mask(
                tumor_mask,
                radius=self.min_distance_to_tumor,
            )

            n_insert = self._sample_num_insertions()
            for _ in range(n_insert):
                donor_image, donor_masks = self._sample_donor(device=device)
                donor_image, donor_masks = self._augment_donor(donor_image, donor_masks)
                placement = self._find_valid_placement(
                    scalar_label=scalar_label,
                    donor_masks=donor_masks,
                    dilated_tumor=dilated_tumor,
                )
                if placement is None:
                    continue

                z0, z1 = placement["z0"], placement["z1"]
                y0, y1 = placement["y0"], placement["y1"]
                x0, x1 = placement["x0"], placement["x1"]
                src_z0, src_z1 = placement["src_z0"], placement["src_z1"]
                src_y0, src_y1 = placement["src_y0"], placement["src_y1"]
                src_x0, src_x1 = placement["src_x0"], placement["src_x1"]
                valid_wt = placement["valid_wt"]
                valid_tc = placement["valid_tc"]
                valid_et = placement["valid_et"]

                target_region = sample_image[:, z0:z1, y0:y1, x0:x1]
                donor_region = donor_image[
                    :,
                    src_z0:src_z1,
                    src_y0:src_y1,
                    src_x0:src_x1,
                ]
                if self.intensity_match:
                    donor_region = self._match_intensity(
                        donor_region=donor_region,
                        target_region=target_region,
                        valid_mask=valid_wt,
                    )
                soft_mask = self._build_soft_mask(valid_wt).unsqueeze(0)
                sample_image[:, z0:z1, y0:y1, x0:x1] = (
                    target_region * (1.0 - soft_mask) + donor_region * soft_mask
                )
                scalar_region = scalar_label[z0:z1, y0:y1, x0:x1].clone()
                scalar_region[valid_wt] = 2
                scalar_region[valid_tc] = 1
                scalar_region[valid_et] = 3
                scalar_label[z0:z1, y0:y1, x0:x1] = scalar_region

                tumor_mask = scalar_label > 0
                dilated_tumor = self._dilate_mask(
                    tumor_mask,
                    radius=self.min_distance_to_tumor,
                )

            image[bidx] = sample_image
            label[bidx] = self._from_scalar_label(
                scalar=scalar_label,
                mode=label_mode,
                dtype=label_dtype,
            )

        batch[self.image_key] = image
        batch[self.label_key] = label
        return batch
