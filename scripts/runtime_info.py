"""Runtime facts for run provenance. Device and threads are resolved at run time, never hardcoded."""
from __future__ import annotations

import os
import platform

import numpy as np


def runtime_info() -> dict:
    """What this process will actually use: MSCR's resolved workers/device, torch threads, library versions."""
    import torch

    from cdd_oran.discovery.mscr import resolve_device, resolve_n_jobs

    return {"mscr_n_jobs": resolve_n_jobs(), "mscr_device": resolve_device(),
            "torch_num_threads": torch.get_num_threads(), "numpy": np.__version__, "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "env": {k: os.environ.get(k) for k in ("MSCR_DEVICE", "MSCR_NUM_THREADS", "OMP_NUM_THREADS")}}
