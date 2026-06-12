"""
Data collator for OCR fine-tuning.

Masking strategy:
  - Padding tokens          → label = -100
  - Image tokens (ID 100270) → label = -100
  - Entire user turn         → label = -100
  - Assistant response only  → label = token_id  (contributes to loss)

Two mask modes (see apply_label_mask / build_keep_token_ids):
  - "all"  : every assistant token contributes (default — text + loc + tags)
  - "bbox" : ONLY <loc_N> and element/structure tags contribute; text content
             is masked. Used for the phase-2 bbox-refinement curriculum, where
             text is already good and we want gradient focused on coordinates.
"""
from __future__ import annotations

import re

IGNORE_INDEX = -100
IMAGE_TOKEN_ID = 100270
DOCTAG_TOKEN_ID = 100327

# Matches docling structural tokens that are single special tokens in the
# granite-docling vocab: <loc_173>, <text>, </text>, <section_header_level_1>,
# <doctag>, </otsl>, <caption>, <list_item>, ... — but NOT chat-role tokens
# like <|start_of_role|> (the '|' fails the class) nor plain text.
_KEEP_TOKEN_RE = re.compile(r"^</?[a-z][a-z0-9_]*>$")


def build_keep_token_ids(tokenizer) -> set[int]:
    """
    Return the set of vocab IDs for loc + element/structure tags (the tokens
    kept in "bbox" mask mode). Scans the full vocab so it adapts to whatever
    loc range / element tags the tokenizer actually defines.
    """
    keep: set[int] = set()
    for tok, tid in tokenizer.get_vocab().items():
        if _KEEP_TOKEN_RE.match(tok):
            keep.add(tid)
    return keep


def find_boundary_idx(input_ids: list[int], boundary_tokens: list[int]) -> int:
    """
    Find the index immediately after the assistant boundary token sequence.

    Args:
        input_ids:       Full token sequence.
        boundary_tokens: Token IDs encoding "<|start_of_role|>assistant<|end_of_role|>".

    Returns:
        Index of the first assistant token (i.e. boundary_end).
        -1 if the boundary sequence is not found.
    """
    n, m = len(input_ids), len(boundary_tokens)
    if m == 0 or n < m:
        return -1
    for i in range(n - m + 1):
        if input_ids[i : i + m] == boundary_tokens:
            return i + m
    return -1


def apply_label_mask(
    input_ids: list[int],
    attention_mask: list[int],
    boundary_end_idx: int,
    image_token_id: int = IMAGE_TOKEN_ID,
    keep_only_ids: set[int] | None = None,
) -> list[int]:
    """
    Build labels from input_ids by masking everything that should not
    contribute to the loss.

    Masked (set to IGNORE_INDEX = -100):
      - All tokens before boundary_end_idx  (user turn + boundary tokens)
      - Any image token (image_token_id) regardless of position
      - Padding tokens  (attention_mask == 0)
      - If keep_only_ids is given: any assistant token whose ID is NOT in the
        set (i.e. "bbox" mode keeps only loc + structure tags, masks text).

    Args:
        input_ids:        Token IDs for a single sequence.
        attention_mask:   1 for real tokens, 0 for padding.
        boundary_end_idx: Index where assistant content starts.
        image_token_id:   Token ID to always mask (default 100270).
        keep_only_ids:    If set, assistant tokens not in it are masked.

    Returns:
        List of int labels, same length as input_ids.
    """
    labels = []
    for i, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
        if mask == 0 or token_id == image_token_id or i < boundary_end_idx:
            labels.append(IGNORE_INDEX)
        elif keep_only_ids is not None and token_id not in keep_only_ids:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(token_id)
    return labels
