"""Compatibility exports for the SnowGym neural executor.

The canonical implementation lives in :mod:`snowgym_training.executor.model`.
This module remains to preserve existing imports and checkpoint tooling.
"""

from .executor.model import *  # noqa: F403
from .executor.model import __all__
