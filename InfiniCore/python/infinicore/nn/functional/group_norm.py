from infinicore.lib import _infinicore
from infinicore.tensor import Tensor, ones, zeros


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Apply GroupNorm to an NCHW tensor."""
    if len(input.shape) != 4:
        raise ValueError(f"group_norm expects NCHW input, got {input.shape}")
    channels = input.shape[1]
    if num_groups <= 0 or channels % num_groups:
        raise ValueError(
            f"channels ({channels}) must be divisible by num_groups ({num_groups})"
        )
    if weight is None:
        weight = ones(
            (channels,), dtype=input.dtype, device=input.device
        )
    if bias is None:
        bias = zeros(
            (channels,), dtype=input.dtype, device=input.device
        )
    if tuple(weight.shape) != (channels,) or tuple(bias.shape) != (channels,):
        raise ValueError("group_norm weight and bias must have shape [channels]")

    if out is None:
        return Tensor(
            _infinicore.group_norm(
                input._underlying,
                num_groups,
                weight._underlying,
                bias._underlying,
                eps,
            )
        )

    _infinicore.group_norm_(
        out._underlying,
        input._underlying,
        num_groups,
        weight._underlying,
        bias._underlying,
        eps,
    )
    return out
