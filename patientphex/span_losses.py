"""Optional training-only objectives for exact overlapping span boundaries."""

from __future__ import annotations

import math
from typing import Any


def boundary_ranking_loss(logits: Any, targets: Any, loss_mask: Any, margin: float = 1.0) -> Any:
    """Rank each positive above its hardest incorrect same-start/end span.

    Inputs have shape ``[batch, start_token, end_minus_start, label]``. For each
    supervised positive label, compare its logit with the largest logit in the
    SAME label channel among spans sharing its start OR end token. Negative
    candidates must be valid/supervised in every channel and cannot be gold in
    ANY channel. This excludes nested positives, correctly bounded NO spans,
    padding/special tokens, out-of-range ends, and all ignored annotations.

    Return the mean ``relu(margin + negative_logit - positive_logit)`` over
    positives with at least one eligible negative. There is one hardest negative
    per positive, regardless of candidate count. The result remains connected
    to logits when there are no eligible pairs, allowing ordinary backward().
    No validation labels or decoded predictions are needed by this objective.
    """
    import torch

    if logits.ndim != 4 or targets.shape != logits.shape or loss_mask.shape != logits.shape:
        raise ValueError("logits, targets, and loss_mask must have the same four-dimensional shape")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if any(size < 1 for size in logits.shape):
        raise ValueError("All span tensor dimensions must be nonempty")
    scores = logits.float()
    supervised = loss_mask.bool()
    positive = targets.gt(0) & supervised
    positions = positive.nonzero(as_tuple=False)
    if positions.shape[0] == 0:
        return scores.sum() * 0.0

    batch, start, delta, label = positions.unbind(dim=1)
    _, length, width, _ = scores.shape
    widths = torch.arange(width, device=scores.device)
    token_starts = torch.arange(length, device=scores.device)
    legal_ends = token_starts[:, None] + widths[None, :] < length
    negative = supervised.all(dim=-1) & ~targets.gt(0).any(dim=-1) & legal_ends.unsqueeze(0)

    same_start_scores = scores[batch[:, None], start[:, None], widths[None, :], label[:, None]]
    same_start_valid = negative[batch[:, None], start[:, None], widths[None, :]]

    ends = start + delta
    other_starts = ends[:, None] - widths[None, :]
    in_bounds = (other_starts >= 0) & (other_starts < length)
    safe_starts = other_starts.clamp(0, length - 1)
    same_end_scores = scores[batch[:, None], safe_starts, widths[None, :], label[:, None]]
    same_end_valid = in_bounds & negative[batch[:, None], safe_starts, widths[None, :]]

    candidates = torch.cat((same_start_scores, same_end_scores), dim=1)
    valid = torch.cat((same_start_valid, same_end_valid), dim=1)
    has_negative = valid.any(dim=1)
    hardest = candidates.masked_fill(~valid, -torch.inf).max(dim=1).values
    positive_scores = scores[batch, start, delta, label]
    # Replace missing maxima before arithmetic to avoid inf * 0 and its gradients.
    hardest = torch.where(has_negative, hardest, positive_scores)
    losses = torch.relu(margin + hardest - positive_scores)
    return (losses * has_negative).sum() / has_negative.sum().clamp_min(1)
