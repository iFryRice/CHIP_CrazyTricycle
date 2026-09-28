"""Create auditable span supervision without changing source text or offsets.

This module has no model-library dependency. Callers provide a fast tokenizer and
split documents before calling it; no dictionary or label vocabulary is learned.
Token span endpoints are inclusive, character span endpoints are exclusive.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


LABEL_NAMES = ("positive", "NO")
ISSUE_NAMES = (
    "invalid_entity_span", "entity_not_in_passage", "text_mismatch",
    "exact_alignment_failed", "span_too_long", "no_window_coverage",
)


def _passage_windows(
    document: dict[str, Any], paragraph_index: int, tokenizer: Any,
    max_length: int, stride: int,
) -> list[dict[str, Any]]:
    passage = document["full_text"][paragraph_index]
    text, origin = passage["text"], passage["offset"]
    if not isinstance(text, str) or type(origin) is not int or origin < 0:
        raise ValueError(f"Invalid paragraph in {document['pmc_id']}:{paragraph_index}")
    encoded = tokenizer(
        text, truncation=True, max_length=max_length, stride=stride,
        return_offsets_mapping=True, return_overflowing_tokens=True,
        return_special_tokens_mask=True, padding=False,
    )
    required = ("input_ids", "attention_mask", "offset_mapping", "special_tokens_mask")
    if any(key not in encoded for key in required):
        raise ValueError("Tokenizer must return IDs, attention, offsets and special-token masks")
    number = len(encoded["input_ids"])
    if number == 0 or any(len(encoded[key]) != number for key in required):
        raise ValueError("Tokenizer returned inconsistent overflow window counts")
    if "token_type_ids" in encoded and len(encoded["token_type_ids"]) != number:
        raise ValueError("Tokenizer returned inconsistent token-type window counts")
    windows = []
    for index in range(number):
        ids = list(encoded["input_ids"][index])
        attention = list(encoded["attention_mask"][index])
        special = list(encoded["special_tokens_mask"][index])
        offsets = list(encoded["offset_mapping"][index])
        if not (len(ids) == len(attention) == len(special) == len(offsets)):
            raise ValueError("Tokenizer returned inconsistent token/offset lengths")
        if len(ids) > max_length:
            raise ValueError("Tokenizer did not respect max_length")
        global_offsets = []
        previous_start = -1
        for pair, visible, is_special in zip(offsets, attention, special):
            if not visible or is_special:
                global_offsets.append(None)
                continue
            if len(pair) != 2:
                raise ValueError("Tokenizer returned an invalid character offset")
            start, end = pair
            if not (type(start) is int and type(end) is int
                    and 0 <= start < end <= len(text) and start >= previous_start):
                raise ValueError("Tokenizer offsets must index the unchanged paragraph text")
            previous_start = start
            global_offsets.append([origin + start, origin + end])
        window = {
            "pmc_id": document["pmc_id"], "paragraph_index": paragraph_index,
            "paragraph_offset": origin, "window_index": index,
            "input_ids": ids, "attention_mask": attention,
            "special_tokens_mask": special, "offset_mapping": global_offsets,
            "span_labels": [],
        }
        if "token_type_ids" in encoded:
            token_types = list(encoded["token_type_ids"][index])
            if len(token_types) != len(ids):
                raise ValueError("Tokenizer returned inconsistent token-type lengths")
            window["token_type_ids"] = token_types
        windows.append(window)
    return windows


def build_span_windows(
    documents: Iterable[dict[str, Any]], tokenizer: Any,
    max_length: int = 384, stride: int = 128, max_span_width: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return tokenizer windows and a complete audit of unencoded gold entities.

    ``offset_mapping`` contains global ``[start, end)`` pairs, or ``None`` for
    special/padding tokens. ``span_labels`` holds inclusive ``start_token`` and
    ``end_token``, two-bit ``labels`` in ``LABEL_NAMES`` order, exact source
    ``offset/length/text``, raw ``identifiers``, and per-entity ``annotations``.
    The latter retain ``entity_index/identifier/note`` without ontology repair.

    Every fully represented gold span is labelled in every containing window.
    Same-span annotations merge into one multi-label target; overlapping and
    crossing spans remain separate targets. Only the exact note ``NO`` negates.
    Audit issue lists are intentionally exhaustive, not truncated examples.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with source offset mappings is required")
    if type(max_length) is not int or max_length < 3:
        raise ValueError("max_length must be an integer of at least three")
    if type(stride) is not int or stride < 0:
        raise ValueError("stride must be a nonnegative integer")
    if type(max_span_width) is not int or max_span_width < 1:
        raise ValueError("max_span_width must be a positive integer")
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = max_length - specials
    if capacity < 1 or stride >= capacity:
        raise ValueError("stride must be smaller than the content-token window capacity")

    counts: Counter[str] = Counter()
    issues: dict[str, list[dict[str, Any]]] = {key: [] for key in ISSUE_NAMES}
    unencoded = []
    windows = []
    seen_documents = set()

    def record_issue(reason: str, metadata: dict[str, Any], **details: Any) -> None:
        issues[reason].append({**metadata, **details})

    for document in documents:
        identifier = document["pmc_id"]
        if identifier in seen_documents:
            raise ValueError(f"Duplicate document identifier: {identifier}")
        seen_documents.add(identifier)
        counts["documents"] += 1
        passages = document["full_text"]
        paragraph_windows = []
        span_indices = []
        for paragraph_index in range(len(passages)):
            current = _passage_windows(document, paragraph_index, tokenizer, max_length, stride)
            paragraph_windows.append(current)
            # One mutable index per window merges same-span gold annotations.
            span_indices.append([{} for _ in current])
            windows.extend(current)
            counts["paragraphs"] += 1
            counts["windows"] += len(current)
            counts["empty_windows"] += sum(not any(w["offset_mapping"]) for w in current)

        for entity_index, entity in enumerate(document.get("entities", [])):
            counts["entities"] += 1
            start, length = entity.get("offset"), entity.get("length")
            metadata = {
                "pmc_id": identifier, "entity_index": entity_index,
                "offset": start, "length": length, "text": entity.get("text"),
                "identifier": entity.get("identifier"), "note": entity.get("note"),
            }
            reasons = []
            if (type(start) is not int or start < 0 or type(length) is not int
                    or length < 1 or not isinstance(entity.get("text"), str)):
                reasons.append("invalid_entity_span")
                record_issue(reasons[-1], metadata)
                unencoded.append({**metadata, "reasons": reasons})
                continue
            end = start + length
            containing = [
                i for i, passage in enumerate(passages)
                if passage["offset"] <= start
                and end <= passage["offset"] + len(passage["text"])
            ]
            matching = [
                i for i in containing
                if passages[i]["text"][start - passages[i]["offset"]:end - passages[i]["offset"]]
                == entity["text"]
            ]
            if not matching:
                reason = "text_mismatch" if containing else "entity_not_in_passage"
                record_issue(reason, metadata)
                unencoded.append({**metadata, "reasons": [reason]})
                continue

            candidates = []
            all_offsets = set()
            start_aligned = end_aligned = char_covered = False
            widths = []
            for paragraph_index in matching:
                for window_index, window in enumerate(paragraph_windows[paragraph_index]):
                    valid = [(i, pair) for i, pair in enumerate(window["offset_mapping"]) if pair is not None]
                    all_offsets.update(tuple(pair) for _, pair in valid)
                    starts = [i for i, pair in valid if pair[0] == start]
                    ends = [i for i, pair in valid if pair[1] == end]
                    start_aligned |= bool(starts)
                    end_aligned |= bool(ends)
                    char_covered |= bool(valid and valid[0][1][0] <= start and end <= valid[-1][1][1])
                    for token_start in starts:
                        for token_end in ends:
                            selected = window["offset_mapping"][token_start:token_end + 1]
                            if token_end < token_start or any(
                                pair is None or pair[0] < start or pair[1] > end
                                for pair in selected
                            ):
                                continue
                            width = token_end - token_start + 1
                            widths.append(width)
                            if width <= max_span_width:
                                candidates.append((paragraph_index, window_index, token_start, token_end))

            if not candidates:
                if not (start_aligned and end_aligned):
                    reasons.append("exact_alignment_failed")
                    record_issue(reasons[-1], metadata, start_aligned=start_aligned, end_aligned=end_aligned)
                estimated_width = sum(start <= left and right <= end for left, right in all_offsets)
                span_width = min(widths) if widths else estimated_width
                if start_aligned and end_aligned and span_width > max_span_width:
                    reasons.append("span_too_long")
                    record_issue(reasons[-1], metadata, token_width=span_width, max_span_width=max_span_width)
                if not char_covered:
                    reasons.append("no_window_coverage")
                    record_issue(reasons[-1], metadata)
                if not reasons:
                    # A tokenizer may use inconsistent offsets between overflow
                    # windows. Never turn such a target silently into a negative.
                    reasons.append("exact_alignment_failed")
                    record_issue(reasons[-1], metadata, start_aligned=start_aligned, end_aligned=end_aligned)
                unencoded.append({**metadata, "reasons": reasons})
                continue

            counts["encoded_entities"] += 1
            label_index = 1 if entity.get("note") == "NO" else 0
            counts["negative_entities" if label_index else "positive_entities"] += 1
            for paragraph_index, window_index, token_start, token_end in candidates:
                window = paragraph_windows[paragraph_index][window_index]
                indexed = span_indices[paragraph_index][window_index]
                key = token_start, token_end
                if key not in indexed:
                    indexed[key] = {
                        "start_token": token_start, "end_token": token_end,
                        "labels": [0, 0], "offset": start, "length": length,
                        "text": entity["text"], "identifiers": [], "annotations": [],
                    }
                    window["span_labels"].append(indexed[key])
                span = indexed[key]
                span["labels"][label_index] = 1
                if entity["identifier"] not in span["identifiers"]:
                    span["identifiers"].append(entity["identifier"])
                span["annotations"].append({
                    "entity_index": entity_index, "identifier": entity["identifier"],
                    "note": entity.get("note"),
                })
                counts["entity_window_assignments"] += 1

    for window in windows:
        window["span_labels"].sort(key=lambda span: (span["start_token"], span["end_token"]))
    counts["unencoded_entities"] = len(unencoded)
    counts["span_window_targets"] = sum(len(window["span_labels"]) for window in windows)
    for key in (
        "documents", "paragraphs", "windows", "empty_windows", "entities",
        "encoded_entities", "positive_entities", "negative_entities", "entity_window_assignments",
    ):
        counts.setdefault(key, 0)
    audit = {
        "label_names": list(LABEL_NAMES),
        "parameters": {"max_length": max_length, "stride": stride, "max_span_width": max_span_width},
        "counts": dict(counts), "issue_counts": {key: len(value) for key, value in issues.items()},
        "issues": issues, "unencoded_entities": unencoded,
    }
    return windows, audit
