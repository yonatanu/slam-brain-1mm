"""Import the dataset loader directly from a source checkout."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
SlamT2 = importlib.import_module("slam_t2_data").SlamT2

__all__ = ["SlamT2"]
