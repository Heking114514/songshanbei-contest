"""Conditioning cache bookkeeping; also works with torch.inference_mode tensors."""

import torch


class MiniMaxH3ConditionCache:
    """Retain snapshots rather than relying on tensor identity or device addresses.

    A revision identifies one exact text/layout pair within one runner. It never
    resets between generations. No generated video, latent, or denoising result
    is stored here.
    """

    def __init__(self) -> None:
        self.generation_id = 0
        self.revision = 0
        self._metadata = None
        self._snapshots: tuple[torch.Tensor, torch.Tensor] | None = None

    def begin_generation(self) -> int:
        self.generation_id += 1
        self._metadata = None
        self._snapshots = None
        return self.generation_id

    def matches(self, text: torch.Tensor, positions: torch.Tensor, metadata) -> bool:
        if self._snapshots is None or self._metadata != metadata:
            return False
        return all(
            saved.shape == current.shape
            and saved.dtype == current.dtype
            and saved.device == current.device
            and torch.equal(saved, current)
            for saved, current in zip(self._snapshots, (text, positions))
        )

    def update(self, text: torch.Tensor, positions: torch.Tensor, metadata) -> None:
        # Commit only once both clones succeed; an allocation failure must not
        # leave a key pointing at snapshots from the previous request.
        snapshots = (text.detach().clone(), positions.detach().clone())
        self._snapshots = snapshots
        self._metadata = metadata
        self.revision += 1
