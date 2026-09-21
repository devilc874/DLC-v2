"""
SOTA / contemporary floating-point compressor baselines for paper benchmarks.

Wrappers try native libraries when installed (pysz, zfpy). Elf and ALP ship
pure-Python reference implementations so comparisons always run without
external research binaries.

Equal-error comparisons for near-lossless methods should use abs_error=8e-6
(DLC's production contract).
"""

from .common import BaselineResult, abs_error_stats
from .sz3_baseline import compress_sz3, available as sz3_available
from .zfp_baseline import compress_zfp, available as zfp_available
from .elf_baseline import compress_elf, available as elf_available
from .alp_baseline import compress_alp, available as alp_available

__all__ = [
    "BaselineResult",
    "abs_error_stats",
    "compress_sz3",
    "compress_zfp",
    "compress_elf",
    "compress_alp",
    "sz3_available",
    "zfp_available",
    "elf_available",
    "alp_available",
    "probe_availability",
]


def probe_availability():
    """Return {name: bool} — all baselines always runnable; note native where useful."""
    from . import sz3_baseline, zfp_baseline
    return {
        "SZ3": True,
        "ZFP": True,
        "Elf": True,
        "ALP": True,
        "SZ3_native": sz3_baseline.native_available(),
        "ZFP_native": zfp_baseline.native_available(),
    }
