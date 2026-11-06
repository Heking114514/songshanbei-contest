from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

module_path = Path.home() / "Projects/InfiniLM/examples/minimax_h3.py"
spec = importlib.util.spec_from_file_location("minimax_h3_example", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {module_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

valid = {
    "videos": np.zeros((1, 124, 768, 1344, 3), dtype=np.uint8),
    "audio": np.zeros((1, 2, 165_600), dtype=np.float32),
    "sampling_rate": 32_000,
}
module.validate_generation_output(valid, 124)
print("valid media contract: ok")

invalid = {
    **valid,
    "videos": np.zeros((1, 123, 768, 1344, 3), dtype=np.uint8),
}
try:
    module.validate_generation_output(invalid, 124)
except RuntimeError as exc:
    print("invalid frame count rejected:", exc)
else:
    raise AssertionError("invalid frame count was accepted")
