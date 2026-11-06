from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def snake_beta(
    input: Tensor,
    alpha_log: Tensor,
    beta_log: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Apply the BigVGAN SnakeBeta activation over [batch, channels, length]."""
    if len(input.shape) != 3:
        raise ValueError(f"snake_beta expects [B, C, L] input, got {input.shape}")
    channels = input.shape[1]
    if alpha_log.numel() != channels or beta_log.numel() != channels:
        raise ValueError("snake_beta alpha_log and beta_log must have C elements")

    if out is None:
        return Tensor(
            _infinicore.snake_beta(
                input._underlying,
                alpha_log._underlying,
                beta_log._underlying,
            )
        )
    _infinicore.snake_beta_(
        out._underlying,
        input._underlying,
        alpha_log._underlying,
        beta_log._underlying,
    )
    return out
