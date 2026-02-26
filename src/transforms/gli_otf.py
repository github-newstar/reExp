from __future__ import annotations

from typing import Sequence

import torch
from monai.transforms import MapTransform


class RandGliPseudoInsertd(MapTransform):
    """
    Phase-A on-the-fly pseudo tumor insertion for BraTS training.

    This module is a lightweight placeholder for GliGAN-based augmentation.
    It injects synthetic ellipsoidal lesions directly on cropped training
    patches, preserving the project data contract:
    - input keys: image, label, case_id
    - output keys unchanged (image, label, case_id)
    """

    def __init__(
        self,
        keys,
        prob: float = 0.6,
        max_insertions: int = 2,
        second_insertion_prob: float = 0.4,
        lesion_scale_range: Sequence[float] = (0.3, 0.8),
        class_probs: Sequence[float] = (0.3, 0.4, 0.3),
        intensity_delta_range: Sequence[float] = (-0.10, 0.20),
        noise_std: float = 0.03,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if not (0.0 <= float(prob) <= 1.0):
            raise ValueError(f"prob must be in [0,1], got {prob}")
        if int(max_insertions) < 1:
            raise ValueError(f"max_insertions must be >= 1, got {max_insertions}")
        if not (0.0 <= float(second_insertion_prob) <= 1.0):
            raise ValueError(
                f"second_insertion_prob must be in [0,1], got {second_insertion_prob}"
            )
        if len(lesion_scale_range) != 2:
            raise ValueError("lesion_scale_range must have 2 values: [low, high]")
        scale_low, scale_high = float(lesion_scale_range[0]), float(lesion_scale_range[1])
        if scale_low <= 0 or scale_high <= 0 or scale_low > scale_high:
            raise ValueError(
                f"invalid lesion_scale_range={lesion_scale_range}, require 0<low<=high"
            )
        if len(class_probs) != 3:
            raise ValueError("class_probs must have 3 values for labels [1,2,3]")
        probs = torch.as_tensor(class_probs, dtype=torch.float32)
        if torch.any(probs < 0):
            raise ValueError(f"class_probs must be non-negative, got {class_probs}")
        if float(probs.sum().item()) <= 0:
            raise ValueError(f"class_probs sum must be > 0, got {class_probs}")
        if len(intensity_delta_range) != 2:
            raise ValueError("intensity_delta_range must have 2 values: [low, high]")
        delta_low = float(intensity_delta_range[0])
        delta_high = float(intensity_delta_range[1])
        if delta_low > delta_high:
            raise ValueError(
                f"invalid intensity_delta_range={intensity_delta_range}, require low<=high"
            )
        if float(noise_std) < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")

        self.prob = float(prob)
        self.max_insertions = int(max_insertions)
        self.second_insertion_prob = float(second_insertion_prob)
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.class_probs = probs / probs.sum()
        self.delta_low = delta_low
        self.delta_high = delta_high
        self.noise_std = float(noise_std)

    @staticmethod
    def _to_scalar_label(label: torch.Tensor):
        """
        Convert label to scalar BraTS map in {0,1,2,3}.
        Supports:
        - [D,H,W]
        - [1,D,H,W]
        - [3,D,H,W] in [TC,WT,ET]
        """
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
        raise ValueError(f"Unsupported label shape for pseudo Gli insertion: {tuple(label.shape)}")

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
    def _sample_center_from_healthy(scalar: torch.Tensor):
        healthy = torch.nonzero(scalar == 0, as_tuple=False)
        if healthy.numel() == 0:
            return None
        idx = int(torch.randint(low=0, high=healthy.shape[0], size=(1,)).item())
        return healthy[idx]

    def _sample_num_insertions(self) -> int:
        n = 1
        while n < self.max_insertions:
            if float(torch.rand(1).item()) < self.second_insertion_prob:
                n += 1
            else:
                break
        return n

    def _sample_class(self) -> int:
        # class ids in BraTS scalar label map.
        class_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        idx = int(torch.multinomial(self.class_probs, num_samples=1).item())
        return int(class_ids[idx].item())

    def _insert_one_lesion(self, image: torch.Tensor, scalar_label: torch.Tensor) -> bool:
        center = self._sample_center_from_healthy(scalar_label)
        if center is None:
            return False

        d, h, w = (int(x) for x in scalar_label.shape)
        cz, cy, cx = (int(center[0].item()), int(center[1].item()), int(center[2].item()))

        # Radius is proportional to patch size with random scaling.
        scale = float(torch.empty(1).uniform_(self.scale_low, self.scale_high).item())
        base_r = max(2, int(round(min(d, h, w) * 0.08 * scale)))
        rz = max(1, int(round(base_r * float(torch.empty(1).uniform_(0.7, 1.2).item()))))
        ry = max(1, int(round(base_r * float(torch.empty(1).uniform_(0.7, 1.2).item()))))
        rx = max(1, int(round(base_r * float(torch.empty(1).uniform_(0.7, 1.2).item()))))

        z0, z1 = max(0, cz - rz), min(d, cz + rz + 1)
        y0, y1 = max(0, cy - ry), min(h, cy + ry + 1)
        x0, x1 = max(0, cx - rx), min(w, cx + rx + 1)

        zz = torch.arange(z0, z1, device=scalar_label.device, dtype=torch.float32) - float(cz)
        yy = torch.arange(y0, y1, device=scalar_label.device, dtype=torch.float32) - float(cy)
        xx = torch.arange(x0, x1, device=scalar_label.device, dtype=torch.float32) - float(cx)
        ellipsoid = (
            (zz[:, None, None] / max(float(rz), 1.0)) ** 2
            + (yy[None, :, None] / max(float(ry), 1.0)) ** 2
            + (xx[None, None, :] / max(float(rx), 1.0)) ** 2
        ) <= 1.0

        target_region = scalar_label[z0:z1, y0:y1, x0:x1]
        valid_mask = ellipsoid & (target_region == 0)
        inserted_voxels = int(valid_mask.sum().item())
        if inserted_voxels < 8:
            return False

        class_id = self._sample_class()
        target_region = target_region.clone()
        target_region[valid_mask] = class_id
        scalar_label[z0:z1, y0:y1, x0:x1] = target_region

        # Pseudo-GliGAN intensity synthesis:
        # local additive bias + Gaussian noise within inserted lesion.
        img_region = image[:, z0:z1, y0:y1, x0:x1].clone()
        for c in range(img_region.shape[0]):
            delta = float(torch.empty(1).uniform_(self.delta_low, self.delta_high).item())
            # ET lesions are usually brighter in T1ce-like channel (index 1).
            if class_id == 3 and c == 1:
                delta *= 1.5
            noise = torch.randn_like(img_region[c]) * self.noise_std
            img_region[c] = torch.where(valid_mask, img_region[c] + delta + noise, img_region[c])
        image[:, z0:z1, y0:y1, x0:x1] = img_region
        return True

    def __call__(self, data):
        output = dict(data)
        if float(torch.rand(1).item()) >= self.prob:
            return output

        image = torch.as_tensor(output["image"]).clone()
        label = torch.as_tensor(output["label"]).clone()

        if image.ndim != 4:
            raise ValueError(
                f"RandGliPseudoInsertd expects image [C,D,H,W], got {tuple(image.shape)}"
            )

        label_dtype = label.dtype
        scalar_label, label_mode = self._to_scalar_label(label)

        n_insert = self._sample_num_insertions()
        for _ in range(n_insert):
            self._insert_one_lesion(image=image, scalar_label=scalar_label)

        output["image"] = image
        output["label"] = self._from_scalar_label(
            scalar=scalar_label,
            mode=label_mode,
            dtype=label_dtype,
        )
        return output
