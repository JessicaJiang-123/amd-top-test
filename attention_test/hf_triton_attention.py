"""
HF Triton Attention - uses SGLang extend_attention_fwd_unified for
true on-policy FSDP-side attention alignment.

Uses the same kernel as SGLang's deterministic inference extend path
(prefix_len=0, sequential kv_indices), so training-side attention is
numerically identical to inference-side extend.
"""
from typing import Optional

import torch

_extend_attention_fwd_unified = None


def _get_extend_attention_fwd_unified():
    global _extend_attention_fwd_unified
    if _extend_attention_fwd_unified is None:
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd_unified,
        )
        _extend_attention_fwd_unified = extend_attention_fwd_unified
    return _extend_attention_fwd_unified


def triton_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    HF AttentionInterface signature.
    query/key/value: (B, num_heads, S, head_dim), GQA: key/value may have fewer heads.
    Returns (attn_output, None), attn_output shape (B, S, num_heads, head_dim).
    """
    extend_attention_fwd_unified = _get_extend_attention_fwd_unified()

    B, H, S, D = query.shape
    kv_heads = key.shape[1]
    device = query.device

    # Reshape to varlen format: [B*S, num_heads, head_dim]
    q = query.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
    k = key.permute(0, 2, 1, 3).reshape(B * S, kv_heads, D).contiguous()
    v = value.permute(0, 2, 1, 3).reshape(B * S, kv_heads, D).contiguous()
    o = torch.empty_like(q)

    total_tokens = B * S

    # qo_indptr: [B+1], each sequence has exactly S tokens
    qo_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * S

    # No prefix cache in teacher-forcing: all KV tokens are extend tokens.
    # kv_indptr == qo_indptr, kv_indices == sequential [0, 1, ..., total_tokens-1].
    kv_indptr = qo_indptr.clone()
    kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int64)

    # prefix_lens = 0 for every sequence (pure prefill, no radix cache)
    prefix_lens = torch.zeros(B, device=device, dtype=torch.int32)

    extend_attention_fwd_unified(
        q,
        o,
        k,
        v,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        max_len_extend=S,
        is_causal=True,
    )

    attn_output = o.reshape(B, S, H, D)
    return attn_output, None