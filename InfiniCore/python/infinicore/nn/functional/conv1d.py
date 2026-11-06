from infinicore.lib import _infinicore
from infinicore.tensor import Tensor, zeros


def _single(value: int | tuple[int]) -> int:
    if isinstance(value, int):
        return value
    if len(value) != 1:
        raise ValueError("conv1d parameters must be scalars or length-1 sequences")
    return int(value[0])


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
    dilation: int | tuple[int] = 1,
    groups: int = 1,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Apply a 1D convolution to an NCL tensor."""
    if len(input.shape) != 3 or len(weight.shape) != 3:
        raise ValueError("conv1d expects 3D input and weight")
    if input.shape[1] % groups:
        raise ValueError("input channels must be divisible by groups")
    output_channels = weight.shape[0]
    if bias is None:
        bias = zeros(
            (output_channels,), dtype=input.dtype, device=input.device
        )
    if bias.numel() != output_channels:
        raise ValueError("conv1d bias must have output_channels elements")

    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)
    if out is None:
        return Tensor(
            _infinicore.conv1d(
                input._underlying,
                weight._underlying,
                bias._underlying,
                stride,
                padding,
                dilation,
                groups,
            )
        )

    _infinicore.conv1d_(
        out._underlying,
        input._underlying,
        weight._underlying,
        bias._underlying,
        stride,
        padding,
        dilation,
        groups,
    )
    return out
