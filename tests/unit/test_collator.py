import pytest

from src.training.collator import (
    _markup_keep_from_offsets,
    apply_label_mask,
    find_boundary_idx,
)


IMAGE_TOKEN_ID = 100270
DOCTAG_TOKEN_ID = 100327
IGNORE_INDEX = -100
PAD_TOKEN_ID = 0

# Placeholder boundary token IDs for "<|start_of_role|>assistant<|end_of_role|>".
# In production these come from the real tokenizer; here we use fixed IDs for determinism.
MOCK_BOUNDARY_TOKENS = [1001, 1002, 1003]


def make_sequence(user_tokens, assistant_tokens, pad_count=0):
    """Build: user_tokens + MOCK_BOUNDARY_TOKENS + assistant_tokens + padding."""
    seq = user_tokens + MOCK_BOUNDARY_TOKENS + assistant_tokens + [PAD_TOKEN_ID] * pad_count
    attn = [1] * (len(user_tokens) + len(MOCK_BOUNDARY_TOKENS) + len(assistant_tokens)) + [0] * pad_count
    boundary_end = len(user_tokens) + len(MOCK_BOUNDARY_TOKENS)
    return seq, attn, boundary_end


class TestApplyLabelMask:
    def test_user_turn_tokens_masked(self):
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10, 20, 30],
            assistant_tokens=[DOCTAG_TOKEN_ID, 50, 60],
        )
        labels = apply_label_mask(seq, attn, boundary_end_idx=boundary_end, image_token_id=IMAGE_TOKEN_ID)
        for i in range(boundary_end):
            assert labels[i] == IGNORE_INDEX, f"token at index {i} (user/boundary) should be masked"

    def test_assistant_tokens_contribute_to_loss(self):
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10, 20],
            assistant_tokens=[DOCTAG_TOKEN_ID, 50, 60],
        )
        labels = apply_label_mask(seq, attn, boundary_end_idx=boundary_end, image_token_id=IMAGE_TOKEN_ID)
        assert labels[boundary_end] == DOCTAG_TOKEN_ID
        assert labels[boundary_end + 1] == 50
        assert labels[boundary_end + 2] == 60

    def test_padding_tokens_masked(self):
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10],
            assistant_tokens=[DOCTAG_TOKEN_ID, 50],
            pad_count=3,
        )
        labels = apply_label_mask(seq, attn, boundary_end_idx=boundary_end, image_token_id=IMAGE_TOKEN_ID)
        for i in range(len(seq) - 3, len(seq)):
            assert labels[i] == IGNORE_INDEX, f"padding token at {i} should be masked"

    def test_image_tokens_in_user_turn_masked(self):
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 20],
            assistant_tokens=[DOCTAG_TOKEN_ID, 50],
        )
        labels = apply_label_mask(seq, attn, boundary_end_idx=boundary_end, image_token_id=IMAGE_TOKEN_ID)
        img_positions = [i for i, t in enumerate(seq) if t == IMAGE_TOKEN_ID]
        for pos in img_positions:
            assert labels[pos] == IGNORE_INDEX

    def test_image_tokens_in_assistant_turn_also_masked(self):
        # Image tokens appearing after the boundary should still be masked
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10],
            assistant_tokens=[DOCTAG_TOKEN_ID, IMAGE_TOKEN_ID, 50],
        )
        labels = apply_label_mask(seq, attn, boundary_end_idx=boundary_end, image_token_id=IMAGE_TOKEN_ID)
        img_pos = seq.index(IMAGE_TOKEN_ID)
        assert labels[img_pos] == IGNORE_INDEX

    def test_first_active_label_is_doctag(self):
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10, 20, 30],
            assistant_tokens=[DOCTAG_TOKEN_ID, 50, 60],
        )
        labels = apply_label_mask(seq, attn, boundary_end_idx=boundary_end, image_token_id=IMAGE_TOKEN_ID)
        active = [l for l in labels if l != IGNORE_INDEX]
        assert len(active) > 0
        assert active[0] == DOCTAG_TOKEN_ID

    def test_all_masked_when_no_assistant_tokens(self):
        seq = [10, 20, 30]
        attn = [1, 1, 1]
        labels = apply_label_mask(seq, attn, boundary_end_idx=3, image_token_id=IMAGE_TOKEN_ID)
        assert all(l == IGNORE_INDEX for l in labels)

    def test_output_length_equals_input_length(self):
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10, 20],
            assistant_tokens=[DOCTAG_TOKEN_ID, 50, 60, 70],
            pad_count=2,
        )
        labels = apply_label_mask(seq, attn, boundary_end_idx=boundary_end, image_token_id=IMAGE_TOKEN_ID)
        assert len(labels) == len(seq)

    def test_returns_list_of_ints(self):
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10],
            assistant_tokens=[DOCTAG_TOKEN_ID, 50],
        )
        labels = apply_label_mask(seq, attn, boundary_end_idx=boundary_end, image_token_id=IMAGE_TOKEN_ID)
        assert isinstance(labels, list)
        assert all(isinstance(l, int) for l in labels)

    def test_has_both_masked_and_active_tokens(self):
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10, 20, 30, 40, 50],
            assistant_tokens=[DOCTAG_TOKEN_ID] + list(range(100, 110)),
        )
        labels = apply_label_mask(seq, attn, boundary_end_idx=boundary_end, image_token_id=IMAGE_TOKEN_ID)
        assert any(l == IGNORE_INDEX for l in labels)
        assert any(l != IGNORE_INDEX for l in labels)


class TestMarkupKeepFromOffsets:
    """Span-based bbox keep: True for tokens inside <...> markup, False for content.
    granite-docling splits <loc_173> into < / loc / _ / 173 / > — all must be kept."""

    def test_keeps_markup_masks_content(self):
        text = "<text><loc_5>hi</text>"
        # <text>=(0,6)  <loc_5>=(6,13)  hi=(13,15)  </text>=(15,22)
        offsets = [(0, 6), (6, 13), (13, 15), (15, 22)]
        assert _markup_keep_from_offsets(text, offsets) == [True, True, False, True]

    def test_loc_coordinate_pieces_kept(self):
        text = "<loc_173>x"
        # < loc _ 173 >  then content 'x'
        offsets = [(0, 1), (1, 4), (4, 5), (5, 8), (8, 9), (9, 10)]
        keep = _markup_keep_from_offsets(text, offsets)
        assert keep[3] is True    # the '173' coordinate token is inside markup
        assert keep[-1] is False  # 'x' content masked

    def test_empty_span_token_masked(self):
        # image/pad tokens often map to (0,0) → no overlap → masked
        assert _markup_keep_from_offsets("<a>", [(0, 0)]) == [False]

    def test_content_between_two_tags(self):
        text = "<a>foo</a>"
        offsets = [(0, 3), (3, 6), (6, 10)]   # <a>  foo  </a>
        assert _markup_keep_from_offsets(text, offsets) == [True, False, True]


class TestFindBoundaryIdx:
    def test_finds_boundary_at_sequence_start(self):
        input_ids = MOCK_BOUNDARY_TOKENS + [DOCTAG_TOKEN_ID, 50]
        idx = find_boundary_idx(input_ids, MOCK_BOUNDARY_TOKENS)
        assert idx == len(MOCK_BOUNDARY_TOKENS)

    def test_finds_boundary_after_user_tokens(self):
        input_ids = [10, 20, 30] + MOCK_BOUNDARY_TOKENS + [DOCTAG_TOKEN_ID, 50]
        idx = find_boundary_idx(input_ids, MOCK_BOUNDARY_TOKENS)
        assert idx == 3 + len(MOCK_BOUNDARY_TOKENS)

    def test_returns_minus_one_when_not_found(self):
        assert find_boundary_idx([10, 20, 30, 40], MOCK_BOUNDARY_TOKENS) == -1

    def test_empty_sequence_returns_minus_one(self):
        assert find_boundary_idx([], MOCK_BOUNDARY_TOKENS) == -1

    def test_sequence_shorter_than_boundary_returns_minus_one(self):
        assert find_boundary_idx([1001], MOCK_BOUNDARY_TOKENS) == -1

    def test_returns_int(self):
        input_ids = [10] + MOCK_BOUNDARY_TOKENS + [DOCTAG_TOKEN_ID]
        idx = find_boundary_idx(input_ids, MOCK_BOUNDARY_TOKENS)
        assert isinstance(idx, int)
