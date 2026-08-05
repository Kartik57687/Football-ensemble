"""
Global seeding.

UNSPECIFIED (docs/unspecified_details.md #1): the paper reports no random seed
and no seeding procedure. Seeds are fixed here purely so that reruns of this
implementation are reproducible; they are not a claim about the original runs.

Note that exact bit-for-bit reproducibility on GPU is not guaranteed even with
seeding, because some cuDNN kernels are non-deterministic. The paper reports
CPU-only training ("no GPU acceleration is utilized"), which is deterministic
under these seeds.
"""

from __future__ import annotations

import os
import random

import numpy as np

from .. import config


def set_global_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, and TensorFlow."""
    seed = config.RANDOM_SEED if seed is None else seed

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass

    return seed
