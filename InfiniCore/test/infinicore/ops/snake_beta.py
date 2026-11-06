import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

_TEST_CASES_DATA = [
    ((1, 4, 17), 4),
    ((2, 8, 31), 8),
    ((1, 16, 257), 16),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-2, "rtol": 2e-2},
    infinicore.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    cases = []
    for shape, channels in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor(shape, None, dtype),
                        TensorSpec.from_tensor((channels,), None, dtype),
                        TensorSpec.from_tensor((channels,), None, dtype),
                    ],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="SnakeBeta - OUT_OF_PLACE",
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("snake_beta")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, hidden_states, alpha_log, beta_log, **kwargs):
        alpha = alpha_log.unsqueeze(0).unsqueeze(-1).exp()
        beta = beta_log.unsqueeze(0).unsqueeze(-1).exp()
        return hidden_states + torch.sin(alpha * hidden_states).pow(2) / (beta + 1e-9)

    def infinicore_operator(self, hidden_states, alpha_log, beta_log, **kwargs):
        import infinicore.nn.functional as F

        return F.snake_beta(hidden_states, alpha_log, beta_log)


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
