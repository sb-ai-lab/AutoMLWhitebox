import numpy as np
import os

from .lib.autowoe import AutoWoE
from .lib.report.report import ReportDeco


__all__ = ["AutoWoE", "ReportDeco"]

if os.getenv("DOCUMENTATION_ENV") is None:
    try:
        import importlib.metadata as importlib_metadata
    except ModuleNotFoundError:
        import importlib_metadata

    __version__ = importlib_metadata.version(__name__)

np.random.seed(42)
