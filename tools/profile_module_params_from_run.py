#!/usr/bin/env python3
"""
Profile per-module parameter counts from a saved run_name.

Focus groups:
- DIDC blocks (class name: DIDCBlock)
- Mamba core modules (class name: Mamba)
- Mamba-like blocks (class name contains "Mamba", excluding core Mamba)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _num_params(module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _collect_group(
    model,
    *,
    class_exact: str | None = None,
    class_contains: str | None = None,
    exclude_exact: set[str] | None = None,
    trainable_only: bool = False,
) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    exclude_exact = exclude_exact or set()
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__
        if class_exact is not None and cls != class_exact:
            continue
        if class_contains is not None and class_contains not in cls:
            continue
        if cls in exclude_exact:
            continue
        n = _num_params(mod, trainable_only=trainable_only)
        if n <= 0:
            continue
        rows.append((name if name else "<root>", cls, n))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows


def _print_group(title: str, rows: list[tuple[str, str, int]], show_top: int) -> int:
    total = sum(n for _, _, n in rows)
    print("-" * 80)
    print(f"{title}: {total} ({total / 1e6:.4f} M)")
    if not rows:
        print("  (none)")
        return total
    top_n = max(show_top, 0)
    if top_n > 0:
        for name, cls, n in rows[:top_n]:
            print(f"  {name} [{cls}]: {n} ({n / 1e6:.4f} M)")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DIDC/Mamba params from run_name.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-root", default="saved")
    parser.add_argument("--show-top", type=int, default=20)
    parser.add_argument("--trainable-only", action="store_true")
    args = parser.parse_args()

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    run_dir = ROOT / args.save_root / args.run_name
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Run config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    model = instantiate(cfg.model)

    total_params = _num_params(model, trainable_only=args.trainable_only)
    didc_rows = _collect_group(
        model,
        class_exact="DIDCBlock",
        trainable_only=args.trainable_only,
    )
    mamba_core_rows = _collect_group(
        model,
        class_exact="Mamba",
        trainable_only=args.trainable_only,
    )
    mamba_like_rows = _collect_group(
        model,
        class_contains="Mamba",
        exclude_exact={"Mamba"},
        trainable_only=args.trainable_only,
    )

    print("=" * 80)
    print("Module Parameter Summary")
    print("=" * 80)
    print(f"run_name           : {args.run_name}")
    print(f"config_path        : {cfg_path}")
    print(f"param_scope        : {'trainable' if args.trainable_only else 'all'}")
    print(f"params_total       : {total_params} ({total_params / 1e6:.4f} M)")

    didc_total = _print_group("DIDCBlock total", didc_rows, args.show_top)
    mamba_core_total = _print_group("Mamba(core) total", mamba_core_rows, args.show_top)
    mamba_like_total = _print_group("Mamba-like block total", mamba_like_rows, args.show_top)

    print("-" * 80)
    print(f"didc_ratio_total   : {didc_total / total_params:.6f}" if total_params > 0 else "didc_ratio_total   : 0")
    print(
        f"mamba_core_ratio   : {mamba_core_total / total_params:.6f}"
        if total_params > 0
        else "mamba_core_ratio   : 0"
    )
    print(
        f"mamba_like_ratio   : {mamba_like_total / total_params:.6f}"
        if total_params > 0
        else "mamba_like_ratio   : 0"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
