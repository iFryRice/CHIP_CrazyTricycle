"""Use held-out BIO spans as independent support for acronym mentions."""

import re

from scripts.evaluate_bio_pilot import merge_entities


def revise_entities(entities, raw_spans, candidates, *, veto_acronyms=False, add_mentions=False):
    support = [span for span in raw_spans if span["scores"].get("positive", 0) >= 0.5]
    kept, removed = [], []
    for entity in entities:
        acronym = re.fullmatch(r"[A-Z][A-Z0-9/+\-]{1,7}", entity["text"]) is not None
        overlap = any(max(0, min(entity["offset"] + entity["length"], span["offset"] + span["length"])
                          - max(entity["offset"], span["offset"])) >= min(entity["length"], span["length"]) * 0.5 for span in support)
        if veto_acronyms and acronym and entity.get("note") != "NO" and not overlap:
            removed.append(entity)
        else:
            kept.append(entity)
    if add_mentions:
        output = merge_entities(kept, candidates, 0.8, 0.9, 0.02)
    else:
        output = sorted(kept, key=lambda entity: (entity["offset"], entity["length"]))
    return output, {"removed": removed, "added": [entity for entity in output if entity not in kept]}
