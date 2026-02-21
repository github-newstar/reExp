import numpy as np
import torch
from monai.utils import type_conversion as monai_type_conversion


_PATCHED = False


def _dtype_numpy_to_torch_fallback(dtype):
    np_dtype = np.dtype(dtype)
    mapping = {
        np.dtype(np.bool_): torch.bool,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.complex64): torch.complex64,
        np.dtype(np.complex128): torch.complex128,
    }
    if np_dtype in mapping:
        return mapping[np_dtype]
    raise TypeError(f"Unsupported numpy dtype for torch conversion: {np_dtype!r}")


def patch_monai_numpy_dtype_compat():
    """
    Patch MONAI dtype conversion for environments where torch.from_numpy(np.empty(...))
    may fail with errors like:
      - TypeError: expected np.ndarray (got numpy.ndarray)
      - RecursionError while rendering numpy dtype
    """
    global _PATCHED
    if _PATCHED:
        return

    original = monai_type_conversion.dtype_numpy_to_torch

    def patched(dtype):
        try:
            return original(dtype)
        except (TypeError, RecursionError):
            return _dtype_numpy_to_torch_fallback(dtype)

    monai_type_conversion.dtype_numpy_to_torch = patched
    _PATCHED = True
