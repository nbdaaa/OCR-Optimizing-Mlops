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

# granite-docling tokenizes loc coords as MULTIPLE pieces ("<loc_173>" → "<",
# "loc", "_", "173", ">") — there are NO single <loc_N> vocab tokens. So bbox
# masking can't use a token-id allow-list; it must keep loss on the token SPANS
# that fall inside DocTags markup ("<...>") and mask the natural-language content
# between tags. _MARKUP_RE matches every markup span (element tags + loc tags +
# literal table HTML); chars outside any span are content → masked.
_MARKUP_RE = re.compile(r"<[^>]*>")


def _markup_keep_from_offsets(text: str, offsets) -> list[bool]:
    """Per-token keep flags: True if the token's char span overlaps any <...>
    markup span in `text`, else False (natural-language content). Tokens with an
    empty span (e.g. (0,0) for image/pad) are False."""
    is_markup = bytearray(len(text))
    for m in _MARKUP_RE.finditer(text):
        for i in range(m.start(), m.end()):
            is_markup[i] = 1
    return [bool(e > s and any(is_markup[s:e])) for s, e in offsets]


def build_bbox_label_keep(tokenizer, text: str):
    """
    Tokenize `text` (assistant DocTags, no special tokens) and return
    (input_ids, keep_flags): keep_flags[i] True for markup tokens (element /
    structure tags + loc coordinate spans), False for text content.

    Requires a fast tokenizer (return_offsets_mapping); raises otherwise — the
    caller then falls back to normal all-token masking for that sample.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], _markup_keep_from_offsets(text, enc["offset_mapping"])


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
) -> list[int]:
    """
    Build labels from input_ids by masking everything that should not
    contribute to the loss.

    Masked (set to IGNORE_INDEX = -100):
      - All tokens before boundary_end_idx  (user turn + boundary tokens)
      - Any image token (image_token_id) regardless of position
      - Padding tokens  (attention_mask == 0)

    bbox-mode (mask text content, keep markup) is applied ON TOP of this by the
    collator using build_bbox_label_keep, not here.

    Args:
        input_ids:        Token IDs for a single sequence.
        attention_mask:   1 for real tokens, 0 for padding.
        boundary_end_idx: Index where assistant content starts.
        image_token_id:   Token ID to always mask (default 100270).

    Returns:
        List of int labels, same length as input_ids.
    """
    labels = []
    for i, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
        if mask == 0 or token_id == image_token_id or i < boundary_end_idx:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(token_id)
    return labels
