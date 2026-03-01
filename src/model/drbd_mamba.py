from __future__ import annotations

import inspect
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class EMAVectorQuantizer3D(nn.Module):
    """
    EMA Vector Quantizer for 3D feature maps.

    Input/Output shape: [B, C, D, H, W]
    """

    def __init__(
        self,
        channels: int,
        num_embeddings: int = 256,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_embeddings <= 1:
            raise ValueError(f"num_embeddings must be > 1, got {num_embeddings}")
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if commitment_weight < 0:
            raise ValueError(f"commitment_weight must be >= 0, got {commitment_weight}")

        self.channels = int(channels)
        self.num_embeddings = int(num_embeddings)
        self.decay = float(decay)
        self.eps = float(eps)
        self.commitment_weight = float(commitment_weight)

        embedding = torch.randn(self.num_embeddings, self.channels)
        embedding = embedding / max(self.channels ** 0.5, 1.0)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("embedding_avg", embedding.clone())

    def _quantize(self, flat: torch.Tensor):
        # Distance in float32 for numerical stability under AMP.
        flat_fp32 = flat.float()
        embedding_fp32 = self.embedding.float()
        distances = (
            flat_fp32.pow(2).sum(dim=1, keepdim=True)
            + embedding_fp32.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * flat_fp32 @ embedding_fp32.t()
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding.index_select(0, indices)
        return quantized, indices

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, num_classes=self.num_embeddings).to(flat.dtype)
        cluster_size = one_hot.sum(dim=0)
        embedding_sum = one_hot.t() @ flat

        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=(1.0 - self.decay))
        self.embedding_avg.mul_(self.decay).add_(embedding_sum, alpha=(1.0 - self.decay))

        total_count = self.cluster_size.sum()
        normalized_count = (
            (self.cluster_size + self.eps)
            / (total_count + self.num_embeddings * self.eps)
            * total_count
        )
        normalized_embedding = self.embedding_avg / normalized_count.unsqueeze(1).clamp_min(self.eps)
        self.embedding.copy_(normalized_embedding)

    @staticmethod
    def _perplexity(indices: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        probs = torch.bincount(indices, minlength=num_embeddings).float()
        probs = probs / probs.sum().clamp_min(1.0)
        return torch.exp(-(probs * (probs + 1e-10).log()).sum())

    def forward(self, x: torch.Tensor):
        b, c, d, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                f"Quantizer expected channels={self.channels}, got input channels={c}."
            )
        flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c)
        quantized_flat, indices = self._quantize(flat)

        if self.training:
            self._ema_update(flat.detach(), indices.detach())

        quantized = quantized_flat.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        commit_loss = self.commitment_weight * F.mse_loss(flat, quantized_flat.detach())
        # Straight-through estimator.
        quantized_st = x + (quantized - x).detach()
        perplexity = self._perplexity(indices, self.num_embeddings).to(x.device)
        return quantized_st, commit_loss, perplexity


class DRBDMambaBlock3D(nn.Module):
    """
    DRBD-Mamba block:
    1) 3D locality-preserving sequencing (Morton order)
    2) bidirectional Mamba scanning
    3) learnable gated fusion of forward/reverse streams
    4) inverse Morton mapping + optional EMA vector quantization
    """

    def __init__(
        self,
        channels: int,
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        sequence_impl: str = "mamba1",
        use_quantizer: bool = True,
        num_embeddings: int = 256,
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.channels = int(channels)
        self.sequence_impl = str(sequence_impl).strip().lower()
        if self.sequence_impl not in {"mamba1", "mamba2"}:
            raise ValueError(
                f"sequence_impl must be one of ['mamba1', 'mamba2'], got {sequence_impl!r}"
            )
        self.use_quantizer = bool(use_quantizer)
        self.seq_norm = nn.LayerNorm(self.channels)
        self.bi_mamba = self._build_sequence_layer(
            d_model=self.channels,
            d_state=mamba_state,
            d_conv=mamba_conv,
            expand=mamba_expand,
        )
        # Per-channel fusion gate.
        self.gate_alpha = nn.Parameter(torch.zeros(self.channels))
        self.quantizer = (
            EMAVectorQuantizer3D(
                channels=self.channels,
                num_embeddings=num_embeddings,
                decay=ema_decay,
                commitment_weight=commitment_weight,
            )
            if self.use_quantizer
            else None
        )
        self.post_fuse = nn.Sequential(
            nn.Conv3d(self.channels, self.channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(self.channels),
            nn.SiLU(inplace=True),
        )
        self._morton_cache: Dict[Tuple[int, int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_sequence_layer(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> nn.Module:
        if self.sequence_impl == "mamba1":
            from mamba_ssm import Mamba

            return Mamba(
                d_model=int(d_model),
                d_state=int(d_state),
                d_conv=int(d_conv),
                expand=int(expand),
            )

        try:
            from mamba_ssm import Mamba2
        except Exception as exc:
            raise ImportError(
                "Mamba2 is required but unavailable. "
                "Please install a mamba-ssm version that provides `Mamba2`."
            ) from exc

        # Keep compatibility across mamba-ssm minor API differences.
        signature = inspect.signature(Mamba2.__init__)
        supported = set(signature.parameters.keys())
        kwargs = {
            "d_model": int(d_model),
            "d_state": int(d_state),
            "d_conv": int(d_conv),
            "expand": int(expand),
        }
        kwargs = {key: value for key, value in kwargs.items() if key in supported}
        if "d_model" not in kwargs:
            raise ValueError("Mamba2 signature does not expose required argument `d_model`.")
        return Mamba2(**kwargs)

    @staticmethod
    def _compute_morton_permutation(
        d: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zz, yy, xx = torch.meshgrid(
            torch.arange(d, device=device, dtype=torch.long),
            torch.arange(h, device=device, dtype=torch.long),
            torch.arange(w, device=device, dtype=torch.long),
            indexing="ij",
        )
        z = zz.reshape(-1)
        y = yy.reshape(-1)
        x = xx.reshape(-1)

        max_bits = max(d, h, w).bit_length()
        morton_key = torch.zeros_like(z)
        for bit in range(max_bits):
            morton_key |= ((x >> bit) & 1) << (3 * bit)
            morton_key |= ((y >> bit) & 1) << (3 * bit + 1)
            morton_key |= ((z >> bit) & 1) << (3 * bit + 2)

        permutation = torch.argsort(morton_key)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel(), device=device, dtype=torch.long)
        return permutation, inverse

    def _get_morton_permutation(
        self,
        d: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (int(d), int(h), int(w), str(device))
        if key not in self._morton_cache:
            self._morton_cache[key] = self._compute_morton_permutation(d=d, h=h, w=w, device=device)
        return self._morton_cache[key]

    def forward(self, x: torch.Tensor):
        residual = x
        b, c, d, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                f"DRBDMambaBlock3D expected channels={self.channels}, got input channels={c}."
            )

        permutation, inverse = self._get_morton_permutation(d=d, h=h, w=w, device=x.device)
        tokens = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)
        tokens = tokens.index_select(1, permutation)
        tokens = self.seq_norm(tokens)

        forward_tokens = self.bi_mamba(tokens)
        reverse_tokens = torch.flip(tokens, dims=[1])
        reverse_tokens = self.bi_mamba(reverse_tokens)
        reverse_tokens = torch.flip(reverse_tokens, dims=[1])

        gate = torch.sigmoid(self.gate_alpha).view(1, 1, c)
        fused_tokens = gate * forward_tokens + (1.0 - gate) * reverse_tokens
        fused_tokens = fused_tokens.index_select(1, inverse)
        fused = fused_tokens.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

        commit_loss = x.new_zeros(())
        perplexity = x.new_zeros(())
        if self.quantizer is not None:
            fused, commit_loss, perplexity = self.quantizer(fused)

        out = self.post_fuse(fused)
        out = out + residual
        stats = {
            "commitment_loss": commit_loss,
            "perplexity": perplexity,
        }
        return out, stats
