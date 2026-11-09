"""Batch independent reference attention heads using existing PyTorch SDPA.

This preserves each head's complete query/key sequence and attention equation.
It changes reference execution scheduling, so disclose it with benchmark results.
"""
import functools
import os


def install_reference_head_batch(max_heads=None, min_score_bytes=None):
    import torch.nn.functional as functional
    max_heads = int(os.environ.get("H3_REFERENCE_MAX_HEADS", "4")) if max_heads is None else max_heads
    min_score_bytes = (256 * 1024 ** 2 if min_score_bytes is None else min_score_bytes)
    if max_heads < 1:
        raise ValueError("Reference head batch must be positive")
    original = functional.scaled_dot_product_attention
    if getattr(original, "_h3_reference_head_batch", False):
        return original._h3_original
    logged = set()

    @functools.wraps(original)
    def batched(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, *, scale=None, enable_gqa=False):
        kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p,
                      is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        heads = query.shape[-3] if query.ndim >= 3 else 0
        score_bytes = heads * query.shape[-2] * key.shape[-2] * query.element_size()
        if (query.ndim < 3 or key.ndim != query.ndim or value.ndim != query.ndim
                or enable_gqa or dropout_p != 0.0
                or heads != key.shape[-3] or heads != value.shape[-3]
                or heads <= max_heads or score_bytes <= min_score_bytes):
            return original(query, key, value, **kwargs)
        shape = tuple(query.shape)
        if shape not in logged:
            print("H3_REFERENCE_SDPA_HEAD_BATCH", shape, "max_heads", max_heads,
                  "unbatched_score_bytes", score_bytes, flush=True)
            logged.add(shape)
        outputs = []
        for start in range(0, heads, max_heads):
            end = min(start + max_heads, heads)
            mask = attn_mask
            if mask is not None and mask.ndim >= 3 and mask.shape[-3] == heads:
                mask = mask[..., start:end, :, :]
            kwargs["attn_mask"] = mask
            outputs.append(original(query[..., start:end, :, :],
                                    key[..., start:end, :, :],
                                    value[..., start:end, :, :], **kwargs))
        import torch
        return torch.cat(outputs, dim=-3)

    batched._h3_reference_head_batch = True
    batched._h3_original = original
    functional.scaled_dot_product_attention = batched
    return original
