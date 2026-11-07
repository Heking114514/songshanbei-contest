"""Validate packed input contracts without loading the native extension."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

# Execute the production dataclass only; importing the runner requires the native
# extension, which is not needed to validate tensor shapes and token counts.
source = Path(__file__).parents[2] / "python/infinilm/multimodal/minimax_h3.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
definition = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef)
                  and node.name == "MiniMaxH3TransformerInputs")
namespace = {"dataclass": dataclass, "torch": torch}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
Inputs = namespace["MiniMaxH3TransformerInputs"]


def packed_inputs():
    return Inputs(
        video_hidden_states=torch.zeros(1, 4, 8),
        audio_hidden_states=torch.zeros(1, 2, 4),
        encoder_hidden_states=torch.zeros(1, 3, 6),
        timestep=torch.zeros(1),
        timestep_indices=torch.zeros(9, dtype=torch.int64),
        token_tags=torch.zeros(9, dtype=torch.int64),
        position_ids=torch.zeros(9, 3),
        video_indices=torch.arange(4),
        audio_indices=torch.arange(4, 6),
        text_indices=torch.arange(6, 9),
    )


def test_valid_packed_modalities():
    packed_inputs().validate()


def test_scalar_positions_report_contract_error():
    with pytest.raises(ValueError, match="position_ids must have shape"):
        replace(packed_inputs(), position_ids=torch.tensor(0)).validate()


@pytest.mark.parametrize("name", ["video_indices", "audio_indices", "text_indices"])
def test_missing_modality_index_rejected_before_native_forward(name):
    data = packed_inputs()
    with pytest.raises(ValueError, match="one index per"):
        replace(data, **{name: getattr(data, name)[:-1]}).validate()


def test_packed_sequence_length_must_equal_total_tokens():
    data = replace(packed_inputs(), position_ids=torch.zeros(10, 3),
                   timestep_indices=torch.zeros(10, dtype=torch.int64),
                   token_tags=torch.zeros(10, dtype=torch.int64))
    with pytest.raises(ValueError, match="Packed modality token counts"):
        data.validate()
