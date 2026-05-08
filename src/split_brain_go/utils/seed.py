"""Reproducibility utilities.

Set every source of randomness to a known seed. Call `set_global_seed(seed)`
once at script entry; that's the only knob.

We do *not* enforce full determinism by default because cuDNN deterministic
mode is significantly slower. Pass `deterministic=True` for paper-quality runs.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch (CPU+GPU). Optional cuDNN determinism.

    Args:
        seed: Integer seed. Same seed → same self-play trajectories given the
            same model. (Note: MCTS noise sources are external to this seed
            unless they also use these RNGs.)
        deterministic: If True, force cuDNN to deterministic algorithms and
            disable benchmark mode. Slows training 10–30 % but eliminates
            run-to-run variation. Use for final reproducible reports only.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass
