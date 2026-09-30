"""Filter existing mapped mentions using source-exact LLM span support."""

import re


def filter_entities(entities, raw_spans, scope):
    if scope not in {"acronyms", "short_terms", "all_mapped"}:
        raise ValueError("Unknown LLM corroboration scope.")
    support = [span for span in raw_spans if span["length"] >= 2]
    kept, removed = [], []
    for entity in entities:
        acronym = re.fullmatch(r"[A-Z][A-Z0-9/+\-]{1,7}", entity["text"]) is not None
        short_term = acronym or re.fullmatch(r"[A-Za-z][A-Za-z-]{0,19}", entity["text"]) is not None
        eligible = {"acronyms": acronym, "short_terms": short_term, "all_mapped": True}[scope]
        eligible = eligible and entity.get("note") != "NO" and entity["identifier"] != "-1"
        supported = any(max(0, min(entity["offset"] + entity["length"], span["offset"] + span["length"])
                            - max(entity["offset"], span["offset"])) >= 0.5 * min(entity["length"], span["length"])
                        for span in support)
        if eligible and not supported:
            removed.append(entity)
        else:
            kept.append(entity)
    return sorted(kept, key=lambda entity: (entity["offset"], entity["length"])), {"removed": removed}
