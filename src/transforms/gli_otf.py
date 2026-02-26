from __future__ import annotations

import warnings
from typing import Sequence

import torch
import torch.nn.functional as F
from monai.transforms import MapTransform
from torch.utils.data import get_worker_info

from src.augmentation.gligan_engine import (
    GliGANLabelSampler,
    GliGANTumourConfig,
    GliGANTumourSynthesizer,
)


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


class RandGliGanInsertd(MapTransform):
    """
    Real GliGAN on-the-fly tumour insertion.

    This transform performs dynamic synthesis with pretrained GliGAN generators
    (4 modality-specific SwinUNETR checkpoints) and inserts synthetic lesions
    into healthy regions after crop.
    """

    def __init__(
        self,
        keys,
        prob: float = 0.4,
        max_insertions: int = 2,
        second_insertion_prob: float = 0.35,
        lesion_scale_range: Sequence[float] = (0.35, 0.85),
        class_probs: Sequence[float] = (0.3, 0.4, 0.3),
        class_rewrite_prob: float = 0.5,
        min_insert_voxels: int = 24,
        noise_std: float = 1.0,
        blend_smooth_steps: int = 1,
        generator_image_size: Sequence[int] = (96, 96, 96),
        generator_feature_size: int = 48,
        generator_use_checkpoint: bool = False,
        generator_device: str = "cpu",
        ckpt_t2f: str = "",
        ckpt_t1c: str = "",
        ckpt_t1n: str = "",
        ckpt_t2w: str = "",
        # Legacy path style from official GliGAN repo:
        # <legacy_logdir>/<modality>/weights/generator_<step>.pt
        legacy_logdir: str = "",
        legacy_step_flair: int | None = None,
        legacy_step_t1ce: int | None = None,
        legacy_step_t1: int | None = None,
        legacy_step_t2: int | None = None,
        strict_checkpoints: bool = True,
        use_label_gan: bool = False,
        label_gan_ckpt: str = "",
        label_latent_dim: int = 100,
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
            raise ValueError("lesion_scale_range must be [low, high]")
        if len(class_probs) != 3:
            raise ValueError("class_probs must contain 3 values for classes [1,2,3]")
        if not (0.0 <= float(class_rewrite_prob) <= 1.0):
            raise ValueError(f"class_rewrite_prob must be in [0,1], got {class_rewrite_prob}")
        if int(min_insert_voxels) < 1:
            raise ValueError(f"min_insert_voxels must be >=1, got {min_insert_voxels}")
        if float(noise_std) < 0:
            raise ValueError(f"noise_std must be >=0, got {noise_std}")
        if int(blend_smooth_steps) < 0:
            raise ValueError(f"blend_smooth_steps must be >=0, got {blend_smooth_steps}")
        if len(generator_image_size) != 3:
            raise ValueError("generator_image_size must have 3 dims")

        probs = torch.as_tensor(class_probs, dtype=torch.float32)
        if torch.any(probs < 0) or float(probs.sum().item()) <= 0:
            raise ValueError(f"invalid class_probs={class_probs}")

        self.prob = float(prob)
        self.max_insertions = int(max_insertions)
        self.second_insertion_prob = float(second_insertion_prob)
        self.scale_low = float(lesion_scale_range[0])
        self.scale_high = float(lesion_scale_range[1])
        self.class_probs = probs / probs.sum()
        self.class_rewrite_prob = float(class_rewrite_prob)
        self.min_insert_voxels = int(min_insert_voxels)
        self.noise_std = float(noise_std)
        self.blend_smooth_steps = int(blend_smooth_steps)

        self.generator_image_size = tuple(int(x) for x in generator_image_size)
        self.generator_feature_size = int(generator_feature_size)
        self.generator_use_checkpoint = self._as_bool(generator_use_checkpoint)
        self.generator_device = str(generator_device)

        self.ckpt_t2f = str(ckpt_t2f or "")
        self.ckpt_t1c = str(ckpt_t1c or "")
        self.ckpt_t1n = str(ckpt_t1n or "")
        self.ckpt_t2w = str(ckpt_t2w or "")
        self.legacy_logdir = str(legacy_logdir or "")
        self.legacy_step_flair = legacy_step_flair
        self.legacy_step_t1ce = legacy_step_t1ce
        self.legacy_step_t1 = legacy_step_t1
        self.legacy_step_t2 = legacy_step_t2
        self.strict_checkpoints = self._as_bool(strict_checkpoints)

        self.use_label_gan = self._as_bool(use_label_gan)
        self.label_gan_ckpt = str(label_gan_ckpt or "")
        self.label_latent_dim = int(label_latent_dim)

        self._synthesizer = None
        self._label_sampler = None
        self._warned_cuda_worker = False

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on"}:
                return True
            if v in {"0", "false", "no", "n", "off", ""}:
                return False
        return bool(value)

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
        raise ValueError(f"Unsupported label shape for GliGAN insertion: {tuple(label.shape)}")

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

    def _sample_class_id(self) -> int:
        idx = int(torch.multinomial(self.class_probs, num_samples=1).item())
        return int([1, 2, 3][idx])

    def _resolve_checkpoint_paths(self):
        ckpt_t2f = self.ckpt_t2f
        ckpt_t1c = self.ckpt_t1c
        ckpt_t1n = self.ckpt_t1n
        ckpt_t2w = self.ckpt_t2w
        if self.legacy_logdir:
            root = self.legacy_logdir.rstrip("/")
            if not ckpt_t2f and self.legacy_step_flair is not None:
                ckpt_t2f = f"{root}/flair/weights/generator_{self.legacy_step_flair}.pt"
            if not ckpt_t1c and self.legacy_step_t1ce is not None:
                ckpt_t1c = f"{root}/t1ce/weights/generator_{self.legacy_step_t1ce}.pt"
            if not ckpt_t1n and self.legacy_step_t1 is not None:
                ckpt_t1n = f"{root}/t1/weights/generator_{self.legacy_step_t1}.pt"
            if not ckpt_t2w and self.legacy_step_t2 is not None:
                ckpt_t2w = f"{root}/t2/weights/generator_{self.legacy_step_t2}.pt"
        return ckpt_t2f, ckpt_t1c, ckpt_t1n, ckpt_t2w

    def _runtime_device(self) -> str:
        desired = self.generator_device.lower()
        worker = get_worker_info()
        if worker is not None and desired.startswith("cuda"):
            if not self._warned_cuda_worker:
                warnings.warn(
                    "RandGliGanInsertd is running inside DataLoader worker. "
                    "Falling back generator_device to cpu for stability. "
                    "Use dataloader.num_workers=0 to keep CUDA synthesis.",
                    stacklevel=2,
                )
                self._warned_cuda_worker = True
            return "cpu"
        return self.generator_device

    def _ensure_runtime(self):
        if self._synthesizer is not None:
            return

        ckpt_t2f, ckpt_t1c, ckpt_t1n, ckpt_t2w = self._resolve_checkpoint_paths()
        if not all([ckpt_t2f, ckpt_t1c, ckpt_t1n, ckpt_t2w]):
            message = (
                "GliGAN checkpoints are incomplete. Required: ckpt_t2f, ckpt_t1c, "
                "ckpt_t1n, ckpt_t2w (or legacy_logdir + legacy steps)."
            )
            if self.strict_checkpoints:
                raise ValueError(message)
            warnings.warn(message + " Falling back to no-op.", stacklevel=2)
            return

        self._synthesizer = GliGANTumourSynthesizer(
            GliGANTumourConfig(
                ckpt_t2f=ckpt_t2f,
                ckpt_t1c=ckpt_t1c,
                ckpt_t1n=ckpt_t1n,
                ckpt_t2w=ckpt_t2w,
                feature_size=self.generator_feature_size,
                use_checkpoint=self.generator_use_checkpoint,
                img_size=self.generator_image_size,
                device=self._runtime_device(),
            )
        )
        if self.use_label_gan:
            if not self.label_gan_ckpt:
                raise ValueError("use_label_gan=True but label_gan_ckpt is empty.")
            self._label_sampler = GliGANLabelSampler(
                checkpoint_path=self.label_gan_ckpt,
                device=self._runtime_device(),
                latent_dim=self.label_latent_dim,
            )

    def _generate_ellipsoid_template(self) -> torch.Tensor:
        d, h, w = self.generator_image_size
        scalar = torch.zeros((d, h, w), dtype=torch.long)
        cz = int(torch.randint(int(0.35 * d), max(int(0.35 * d) + 1, int(0.65 * d)), (1,)).item())
        cy = int(torch.randint(int(0.35 * h), max(int(0.35 * h) + 1, int(0.65 * h)), (1,)).item())
        cx = int(torch.randint(int(0.35 * w), max(int(0.35 * w) + 1, int(0.65 * w)), (1,)).item())
        scale = float(torch.empty(1).uniform_(self.scale_low, self.scale_high).item())
        base_r = max(2, int(round(min(d, h, w) * 0.08 * scale)))
        rz = max(1, int(round(base_r * float(torch.empty(1).uniform_(0.7, 1.2).item()))))
        ry = max(1, int(round(base_r * float(torch.empty(1).uniform_(0.7, 1.2).item()))))
        rx = max(1, int(round(base_r * float(torch.empty(1).uniform_(0.7, 1.2).item()))))
        zz = torch.arange(0, d, dtype=torch.float32) - float(cz)
        yy = torch.arange(0, h, dtype=torch.float32) - float(cy)
        xx = torch.arange(0, w, dtype=torch.float32) - float(cx)
        ellipsoid = (
            (zz[:, None, None] / max(float(rz), 1.0)) ** 2
            + (yy[None, :, None] / max(float(ry), 1.0)) ** 2
            + (xx[None, None, :] / max(float(rx), 1.0)) ** 2
        ) <= 1.0
        scalar[ellipsoid] = self._sample_class_id()
        return scalar

    @staticmethod
    def _bbox(mask: torch.Tensor):
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.numel() == 0:
            return None
        mins = coords.min(dim=0).values
        maxs = coords.max(dim=0).values + 1
        return tuple(int(x.item()) for x in mins), tuple(int(x.item()) for x in maxs)

    @staticmethod
    def _rescale_template(template: torch.Tensor, scale: float) -> torch.Tensor:
        if abs(scale - 1.0) < 1e-3:
            return template
        d, h, w = template.shape
        new_d = max(1, int(round(d * scale)))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        out = F.interpolate(
            template.float().unsqueeze(0).unsqueeze(0),
            size=(new_d, new_h, new_w),
            mode="nearest",
        )[0, 0]
        return out.long()

    def _sample_template(self) -> torch.Tensor:
        if self._label_sampler is not None:
            template = self._label_sampler.sample_scalar_label(self.generator_image_size)
        else:
            template = self._generate_ellipsoid_template()

        scale = float(torch.empty(1).uniform_(self.scale_low, self.scale_high).item())
        template = self._rescale_template(template, scale=scale)

        if float(torch.rand(1).item()) < self.class_rewrite_prob:
            class_id = self._sample_class_id()
            nz = template > 0
            template = template.clone()
            template[nz] = class_id
        return template

    def _place_template(
        self,
        current_label: torch.Tensor,
        template: torch.Tensor,
    ) -> torch.Tensor | None:
        template_mask = template > 0
        if int(template_mask.sum().item()) == 0:
            return None
        center = self._sample_center_from_healthy(current_label)
        if center is None:
            return None
        d, h, w = (int(x) for x in current_label.shape)
        td, th, tw = (int(x) for x in template.shape)
        cz, cy, cx = (int(center[0].item()), int(center[1].item()), int(center[2].item()))

        z0_raw = cz - td // 2
        y0_raw = cy - th // 2
        x0_raw = cx - tw // 2
        z1_raw = z0_raw + td
        y1_raw = y0_raw + th
        x1_raw = x0_raw + tw

        z0 = max(0, z0_raw)
        y0 = max(0, y0_raw)
        x0 = max(0, x0_raw)
        z1 = min(d, z1_raw)
        y1 = min(h, y1_raw)
        x1 = min(w, x1_raw)

        src_z0 = max(0, -z0_raw)
        src_y0 = max(0, -y0_raw)
        src_x0 = max(0, -x0_raw)
        src_z1 = src_z0 + (z1 - z0)
        src_y1 = src_y0 + (y1 - y0)
        src_x1 = src_x0 + (x1 - x0)

        if min(z1 - z0, y1 - y0, x1 - x0) <= 0:
            return None

        template_crop = template[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
        target_region = current_label[z0:z1, y0:y1, x0:x1]
        valid = (template_crop > 0) & (target_region == 0)
        if int(valid.sum().item()) < self.min_insert_voxels:
            return None

        insertion = torch.zeros_like(current_label, dtype=torch.long)
        region = insertion[z0:z1, y0:y1, x0:x1]
        region[valid] = template_crop[valid]
        insertion[z0:z1, y0:y1, x0:x1] = region
        return insertion

    def _blend_image(
        self,
        image: torch.Tensor,
        generated: torch.Tensor,
        insertion_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = insertion_mask.float()
        if self.blend_smooth_steps > 0:
            m = mask.unsqueeze(0).unsqueeze(0)
            for _ in range(self.blend_smooth_steps):
                m = F.avg_pool3d(m, kernel_size=3, stride=1, padding=1)
            mask = torch.clamp(m[0, 0], 0.0, 1.0)
        mask = mask.unsqueeze(0)  # [1,D,H,W]
        return image * (1.0 - mask) + generated * mask

    def __call__(self, data):
        output = dict(data)
        if float(torch.rand(1).item()) >= self.prob:
            return output

        image = torch.as_tensor(output["image"]).clone().float()
        label = torch.as_tensor(output["label"]).clone()
        if image.ndim != 4 or image.shape[0] != 4:
            raise ValueError(
                f"RandGliGanInsertd expects image [4,D,H,W], got {tuple(image.shape)}"
            )

        label_dtype = label.dtype
        scalar_label, label_mode = self._to_scalar_label(label)
        self._ensure_runtime()
        if self._synthesizer is None:
            return output

        n_insert = self._sample_num_insertions()
        for _ in range(n_insert):
            template = self._sample_template()
            insertion = self._place_template(current_label=scalar_label, template=template)
            if insertion is None:
                continue
            generated = self._synthesizer.synthesize(
                image=image,
                lesion_scalar=insertion,
                noise_std=self.noise_std,
            )
            insertion_mask = insertion > 0
            image = self._blend_image(image=image, generated=generated, insertion_mask=insertion_mask)
            scalar_label[insertion_mask] = insertion[insertion_mask]

        output["image"] = image
        output["label"] = self._from_scalar_label(
            scalar=scalar_label,
            mode=label_mode,
            dtype=label_dtype,
        )
        return output
