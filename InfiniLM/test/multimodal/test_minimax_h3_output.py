"""Validate the actual Diffusers PIL-batch output before media encoding."""
import ast
from pathlib import Path
from typing import Any
import unittest

import numpy as np
from PIL import Image

SOURCE = Path(__file__).parents[2] / "examples/minimax_h3.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
definition = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "validate_generation_output")
namespace = {"Any": Any, "np": np, "FPS": 24}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), namespace)
validate = namespace["validate_generation_output"]


def output():
    return {
        "videos": [[Image.new("RGB", (64, 64)) for _ in range(22)]],
        "audio": np.zeros((1, 2, 29600), dtype=np.float32),
        "sampling_rate": 32000,
    }


class OutputContractTests(unittest.TestCase):
    def test_diffusers_pil_batch_is_accepted_and_preserved(self):
        result = output()
        frames = result["videos"][0]
        validate(result, 22)
        self.assertIs(result["videos"][0], frames)

    def test_array_batches_are_accepted(self):
        for batch_list in [False, True]:
            with self.subTest(batch_list=batch_list):
                result = output()
                array = np.zeros((22, 64, 64, 3), dtype=np.uint8)
                result["videos"] = [array] if batch_list else array[None]
                validate(result, 22)

    def test_malformed_media_is_rejected(self):
        for failure in [
            "missing_frame", "different_frame_size", "grayscale", "empty_batch",
            "extra_batch", "audio_too_short", "wrong_sample_rate",
        ]:
            with self.subTest(failure=failure):
                result = output()
                if failure == "missing_frame":
                    result["videos"][0].pop()
                elif failure == "different_frame_size":
                    result["videos"][0][0] = Image.new("RGB", (32, 64))
                elif failure == "grayscale":
                    result["videos"][0] = [Image.new("L", (64, 64)) for _ in range(22)]
                elif failure == "empty_batch":
                    result["videos"] = []
                elif failure == "extra_batch":
                    result["videos"] *= 2
                elif failure == "audio_too_short":
                    result["audio"] = result["audio"][:, :, :10]
                elif failure == "wrong_sample_rate":
                    result["sampling_rate"] = 16000
                with self.assertRaises(RuntimeError):
                    validate(result, 22)


if __name__ == "__main__":
    unittest.main(verbosity=2)
