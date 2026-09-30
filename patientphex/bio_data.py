"""Flat BIO supervision and original-offset decoding for phenotype mentions."""

from collections import defaultdict
import math

import numpy as np


LABELS = ("O", "B", "I")


def bio_targets(window, document):
    """Keep longest nonoverlapping gold spans; mask partial/unencoded regions.

    This projection changes training supervision only. Original annotations and
    all evaluation gold, including nested mentions, remain unchanged.
    """
    offsets = window["offset_mapping"]
    targets = [0 if pair is not None else -100 for pair in offsets]
    occupied, selected, omitted = set(), [], []
    spans = sorted(window["span_labels"], key=lambda span: (-(span["end_token"] - span["start_token"]), span["start_token"], span["end_token"]))
    for span in spans:
        tokens = set(range(span["start_token"], span["end_token"] + 1))
        if occupied & tokens:
            omitted.append((span["offset"], span["length"]))
            continue
        occupied.update(tokens)
        targets[span["start_token"]] = 1
        for index in range(span["start_token"] + 1, span["end_token"] + 1):
            targets[index] = 2
        selected.append((span["offset"], span["length"]))
    # A gold mention clipped at an overflow boundary must not teach O labels.
    # The same mask also protects annotation/tokenizer alignment failures.
    visible = [pair for pair in offsets if pair is not None]
    lower = min((pair[0] for pair in visible), default=0)
    upper = max((pair[1] for pair in visible), default=0)
    for entity in document["entities"]:
        start, end = entity["offset"], entity["offset"] + entity["length"]
        if end <= lower or start >= upper:
            continue
        for index, pair in enumerate(offsets):
            if pair is not None and pair[0] < end and start < pair[1] and targets[index] == 0:
                targets[index] = -100
    return targets, {"selected": selected, "omitted_overlaps": omitted}


def add_window_probabilities(accumulator, window, probabilities):
    if len(probabilities) < len(window["offset_mapping"]):
        raise ValueError("Missing token probabilities.")
    key = str(window["pmc_id"]), window["paragraph_index"]
    bucket = accumulator.setdefault(key, {})
    for pair, vector in zip(window["offset_mapping"], probabilities, strict=False):
        if pair is None:
            continue
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all() or (vector < 0).any():
            raise ValueError("Malformed BIO token probabilities.")
        if not np.isclose(vector.sum(), 1.0, atol=1e-4):
            raise ValueError("BIO token probabilities must sum to one.")
        offset = tuple(pair)
        if offset not in bucket:
            bucket[offset] = [vector.copy(), 1]
        else:
            bucket[offset][0] += vector
            bucket[offset][1] += 1


def decode_probabilities(documents, accumulator, minimum_score=0.0):
    """Average overlapping windows before one BIO decode per original paragraph."""
    result = []
    for document in documents:
        spans = []
        for paragraph_index, paragraph in enumerate(document["full_text"]):
            bucket = accumulator.get((str(document["pmc_id"]), paragraph_index), {})
            active = []

            def finish():
                if not active:
                    return
                start, end = active[0][0], active[-1][1]
                origin = paragraph["offset"]
                if not origin <= start < end <= origin + len(paragraph["text"]):
                    raise ValueError("Decoded span escapes its source paragraph.")
                confidence = math.exp(sum(math.log(max(item[2], 1e-12)) for item in active) / len(active))
                if confidence >= minimum_score:
                    spans.append({"offset": start, "length": end - start,
                                  "text": paragraph["text"][start - origin:end - origin],
                                  "labels": ["positive"], "scores": {"positive": confidence}})
                active.clear()

            for (start, end), (summed, count) in sorted(bucket.items()):
                probabilities = summed / count
                label = int(np.argmax(probabilities))
                if label == 0:
                    finish()
                else:
                    if label == 1:
                        finish()
                    # An orphan I is interpreted as a new mention, a deterministic
                    # legal-BIO repair that does not consult answer annotations.
                    active.append((start, end, float(probabilities[label])))
            finish()
        result.append({"pmc_id": document["pmc_id"], "spans": sorted(spans, key=lambda span: (span["offset"], span["length"]))})
    return result
