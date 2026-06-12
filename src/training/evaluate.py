"""
Evaluation metrics for OCR fine-tuning.
"""
from __future__ import annotations

import re


def strip_xml_tags(text: str) -> str:
    """
    Remove all XML/docling tags from text, preserving plain content.

    Strips: <doctag>, </doctag>, <loc_N>, element tags such as
    <text>, </text>, <section_header_level_1>, etc.

    Args:
        text: Raw model output containing docling markup.

    Returns:
        Plain text with all tags removed.
    """
    return re.sub(r"<[^>]+>", "", text)


def compute_cer(prediction: str, ground_truth: str) -> float:
    """
    Compute Character Error Rate (CER).

    CER = edit_distance(prediction, ground_truth) / len(ground_truth)

    Examples:
        >>> compute_cer("hello", "hello")
        0.0
        >>> compute_cer("helo", "hello")   # 1 deletion, len(gt)=5
        0.2
        >>> compute_cer("bello", "hello")  # 1 substitution, len(gt)=5
        0.2
        >>> compute_cer("", "abc")         # 3 insertions, len(gt)=3
        1.0
        >>> compute_cer("hello world", "hello")  # CER > 1.0 khi pred dài hơn gt
        1.2

    Edge cases:
      - Both empty → 0.0
      - Ground truth empty, prediction non-empty → 0.0
      - CER can exceed 1.0 when prediction is longer than ground truth.

    Args:
        prediction:   Model output string.
        ground_truth: Reference string.

    Returns:
        CER as a float >= 0.0.
    """
    if not ground_truth:
        return 0.0

    # Levenshtein distance via DP
    m, n = len(prediction), len(ground_truth)
    dp = list(range(m + 1))

    for j in range(1, n + 1):
        prev = dp[0]
        dp[0] = j
        for i in range(1, m + 1):
            temp = dp[i]
            if prediction[i - 1] == ground_truth[j - 1]:
                dp[i] = prev
            else:
                dp[i] = 1 + min(prev, dp[i], dp[i - 1])
            prev = temp

    return dp[m] / n


def compute_batch_cer(predictions: list[str], ground_truths: list[str]) -> float:
    """
    Compute average CER over a batch.

    Args:
        predictions:   List of model output strings.
        ground_truths: List of reference strings.

    Returns:
        Mean CER across all samples.

    Raises:
        ValueError: If lists are empty or have different lengths.
    """
    if not predictions or not ground_truths:
        raise ValueError("predictions and ground_truths must be non-empty")
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground_truths"
        )
    return sum(compute_cer(p, g) for p, g in zip(predictions, ground_truths)) / len(predictions)


# ── Bounding-box (loc) accuracy ───────────────────────────────────────────────

_LOC_RE = re.compile(r"<loc_(\d+)>")


def extract_locs(text: str) -> list[int]:
    """Return all <loc_N> integer values in document order."""
    return [int(v) for v in _LOC_RE.findall(text)]


def compute_loc_mae(
    predictions: list[str], ground_truths: list[str]
) -> tuple[float, float]:
    """
    Mean Absolute Error (in loc units, 0–500 grid) between predicted and
    ground-truth <loc_N> sequences, plus loc-count coverage.

    Loc values are aligned by position (docling emits elements in reading
    order), comparing up to the shorter of the two sequences per sample. This
    drifts if the model emits a different number of elements — coverage
    (pred_locs / gt_locs) flags how reliable the MAE is.

    Returns:
        (mae, coverage). mae = float("nan") if no loc pairs were comparable.
    """
    total_err = 0.0
    matched = 0
    gt_total = 0
    pred_total = 0
    for pred, gt in zip(predictions, ground_truths):
        pl, gl = extract_locs(pred), extract_locs(gt)
        gt_total += len(gl)
        pred_total += len(pl)
        n = min(len(pl), len(gl))
        for i in range(n):
            total_err += abs(pl[i] - gl[i])
            matched += 1
    mae = (total_err / matched) if matched else float("nan")
    coverage = (pred_total / gt_total) if gt_total else float("nan")
    return mae, coverage
