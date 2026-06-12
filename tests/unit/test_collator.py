import pytest

from src.training.collator import (
    apply_label_mask,
    build_keep_token_ids,
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


class TestBboxMaskMode:
    """keep_only_ids → only loc/structure tokens contribute (text content masked)."""

    def test_only_kept_ids_contribute(self):
        # assistant = [tag(900), loc(910), text(200), text(201), loc(911)]
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10, 20],
            assistant_tokens=[900, 910, 200, 201, 911],
        )
        keep = {900, 910, 911}  # tag + loc tokens; 200/201 are text
        labels = apply_label_mask(
            seq, attn, boundary_end_idx=boundary_end,
            image_token_id=IMAGE_TOKEN_ID, keep_only_ids=keep,
        )
        a = labels[boundary_end:]
        assert a == [900, 910, IGNORE_INDEX, IGNORE_INDEX, 911]

    def test_text_only_assistant_fully_masked(self):
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10],
            assistant_tokens=[200, 201, 202],  # no kept ids
        )
        labels = apply_label_mask(
            seq, attn, boundary_end_idx=boundary_end,
            image_token_id=IMAGE_TOKEN_ID, keep_only_ids={900, 910},
        )
        assert all(l == IGNORE_INDEX for l in labels)

    def test_none_keep_ids_is_backward_compatible(self):
        seq, attn, boundary_end = make_sequence(
            user_tokens=[10], assistant_tokens=[DOCTAG_TOKEN_ID, 50, 60],
        )
        labels = apply_label_mask(seq, attn, boundary_end_idx=boundary_end,
                                  image_token_id=IMAGE_TOKEN_ID, keep_only_ids=None)
        assert labels[boundary_end:] == [DOCTAG_TOKEN_ID, 50, 60]


class TestBuildKeepTokenIds:
    class _FakeTok:
        def __init__(self, vocab):
            self._v = vocab
        def get_vocab(self):
            return self._v

    def test_picks_loc_and_element_tags_only(self):
        vocab = {
            "<loc_0>": 1, "<loc_173>": 2, "<loc_500>": 3,
            "<text>": 4, "</text>": 5, "<section_header_level_1>": 6,
            "<doctag>": 7, "</otsl>": 8,
            "<|start_of_role|>": 9,    # chat role → excluded (has '|')
            "hello": 10, "Ố": 11, " the": 12,   # normal text → excluded
        }
        keep = build_keep_token_ids(self._FakeTok(vocab))
        assert keep == {1, 2, 3, 4, 5, 6, 7, 8}


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
