"""CPU scheduling regression with an independent scalar attention oracle.

No Torch/DTK is required. This checks query partitioning and full-key coverage,
not GPU arithmetic, the native binary or performance.
"""
import importlib.util
import itertools
import math
from pathlib import Path
import random
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class Tensor:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]), len(data[0][0]), len(data[0][0][0]))

    def __getitem__(self, index):
        _, rows, columns = index
        return Tensor([[[row[columns] for row in head[rows]]
                        for head in batch] for batch in self.data])

    def __setitem__(self, index, value):
        _, rows, columns = index
        selected = range(*rows.indices(self.shape[-2]))
        for b, batch in enumerate(self.data):
            for h, head in enumerate(batch):
                for r, original in enumerate(selected):
                    head[original][columns] = value.data[b][h][r]


def scalar_attention(query, key, value, *, dropout_p, is_causal, scale):
    assert dropout_p == 0.0 and not is_causal
    factor = scale if scale is not None else query.shape[-1] ** -0.5
    output = []
    for batch, qbatch in enumerate(query.data):
        output.append([])
        for head, qhead in enumerate(qbatch):
            rows = []
            for qrow in qhead:
                scores = [sum(a * b for a, b in zip(qrow, krow)) * factor
                          for krow in key.data[batch][head]]
                peak = max(scores)
                weights = [math.exp(x - peak) for x in scores]
                denominator = sum(weights)
                rows.append([sum(w * vrow[d] for w, vrow in
                                 zip(weights, value.data[batch][head])) / denominator
                             for d in range(value.shape[-1])])
            output[-1].append(rows)
    return Tensor(output)


def empty_like(tensor):
    b, h, s, d = tensor.shape
    return Tensor([[[[0.0] * d for _ in range(s)] for _ in range(h)] for _ in range(b)])


class QueryScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("attention_policy.py")
        spec = importlib.util.spec_from_file_location("experiment_policy_under_test", path)
        cls.policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.policy)

    def test_odd_tail_multiple_batches_heads_and_full_key_sequence(self):
        rng = random.Random(42)
        def make(length):
            return Tensor([[[[rng.uniform(-1, 1) for _ in range(4)]
                             for _ in range(length)] for _ in range(2)] for _ in range(2)])
        q, k, v = make(7), make(9), make(9)
        for scale, chunk in itertools.product([None, 0.125], [0, 1, 2, 3, 4, 7, 11]):
            with self.subTest(scale=scale, chunk=chunk):
                expected = scalar_attention(q, k, v, dropout_p=0.0,
                                            is_causal=False, scale=scale)
                torch = ModuleType("torch")
                nn = ModuleType("torch.nn")
                functional = ModuleType("torch.nn.functional")
                functional.scaled_dot_product_attention = scalar_attention
                nn.functional = functional
                torch.nn = nn
                torch.empty_like = empty_like
                with patch.dict(sys.modules, {"torch": torch, "torch.nn": nn,
                                              "torch.nn.functional": functional}):
                    actual = self.policy.query_chunk_attention(q, k, v, chunk, scale=scale)
                self.assertEqual(actual.data, expected.data)

    def test_negative_chunk_rejected(self):
        torch = ModuleType("torch")
        nn = ModuleType("torch.nn")
        functional = ModuleType("torch.nn.functional")
        nn.functional = functional
        torch.nn = nn
        with patch.dict(sys.modules, {"torch": torch, "torch.nn": nn,
                                      "torch.nn.functional": functional}):
            with self.assertRaises(ValueError):
                self.policy.query_chunk_attention(None, None, None, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
