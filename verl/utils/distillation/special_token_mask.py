# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Special-token first-occurrence masking for on-policy distillation.

Ported from ``revisiting_opd`` (verl/workers/actor/dp_actor.py:_init_opd_mask_tokens
and _compute_opd_kl_mask). The trick mitigates spurious KL spikes coming
from chat-template / reasoning-marker tokens (e.g. ``<think>``,
``</think>``, ``<|im_end|>``) that may be tokenized differently by the
student and the teacher.
"""

from __future__ import annotations

from typing import Iterable

import torch


def encode_special_token_ids(tokenizer, token_strings: Iterable[str]) -> list[int]:
    """Encode a list of token strings to vocabulary ids.

    Tokens that decompose into multiple sub-tokens contribute every sub-token
    id (mirrors the original implementation, which also masks each sub-token
    individually). Unknown tokens are silently skipped.
    """
    if tokenizer is None:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for token in token_strings:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
        except Exception:
            continue
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                out.append(int(tid))
    return out


def build_first_occurrence_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    token_ids: list[int],
) -> torch.Tensor:
    """Return ``response_mask`` with the first occurrence of any ``token_ids``
    zeroed out per row.

    Args:
        responses: ``(B, T)`` int64 token ids.
        response_mask: ``(B, T)`` boolean / 0-1 mask. Can be a
          ``torch.nested`` tensor (will be padded then masked).
        token_ids: token ids whose first occurrence (per row) is masked.

    Returns:
        Tensor of the same dtype/shape as the (padded) ``response_mask``
        with the targeted positions set to 0.
    """
    if not token_ids:
        return response_mask
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    if responses.is_nested:
        responses = responses.to_padded_tensor(0)
    assert responses.shape == response_mask.shape, (
        f"responses {tuple(responses.shape)} vs response_mask {tuple(response_mask.shape)}"
    )

    out = response_mask.clone()
    # Match the original `revisiting_opd` semantics: for *each* token id, mask
    # the first occurrence per row. (Different ids therefore each consume one
    # masked position when present.)
    rows = torch.arange(responses.size(0), device=responses.device)
    for tid in token_ids:
        matches = responses == int(tid)  # (B, T) bool
        has_any = matches.any(dim=-1)
        if not has_any.any():
            continue
        first_idx = matches.float().argmax(dim=-1)  # (B,)
        sel_rows = rows[has_any]
        sel_cols = first_idx[has_any]
        out[sel_rows, sel_cols] = 0
    return out
