"""Cross-environment discovery methods (E2, E5). Env-specific methods live in ``e1slice``/``e2slice``."""

from cdd_oran.discovery.mscr import (
    MSCR_VERSION,
    MSCRConfig,
    MSCRResult,
    by_declare,
    discover_mscr,
    frozen_config,
)

__all__ = ["MSCR_VERSION", "MSCRConfig", "MSCRResult", "by_declare", "discover_mscr", "frozen_config"]
