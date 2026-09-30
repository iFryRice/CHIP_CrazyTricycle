"""Conservative nested-span additions from frozen neural candidates."""

from copy import deepcopy


def overlap(left, right):
    return left["offset"] < right["offset"] + right["length"] and right["offset"] < left["offset"] + left["length"]


def contains(left, right):
    return left["offset"] <= right["offset"] and right["offset"] + right["length"] <= left["offset"] + left["length"]


def add_nested(document, before_review, reviewed, candidates, *, span_threshold=0.9, semantic_threshold=0.9, margin_threshold=0.02):
    if document.get("entities") or document.get("association"):
        raise ValueError("Nested-span inference requires a blind source document.")
    prior_boundaries = {(e["offset"], e["length"]) for e in before_review}
    ranked = sorted(candidates, key=lambda item: (-item["span_score"] * item["similarity"], item["entity"]["length"],
                                                  item["entity"]["offset"], item["entity"]["identifier"]))
    additions = []
    for item in ranked:
        entity = item["entity"]
        if entity.get("note") == "NO" or not entity["identifier"].startswith("HP:"):
            continue
        if item["span_score"] < span_threshold or (item["source"] == "semantic" and
                (item["similarity"] < semantic_threshold or item["margin"] < margin_threshold)):
            continue
        if (entity["offset"], entity["length"]) in prior_boundaries:
            continue
        anchors = [old for old in reviewed if overlap(entity, old)]
        if not anchors or any(old.get("note") == "NO" or not (contains(entity, old) or contains(old, entity)) for old in anchors):
            continue
        if any(overlap(entity, old) for old in additions):
            continue
        if not any(p["offset"] <= entity["offset"] and entity["offset"] + entity["length"] <= p["offset"] + len(p["text"])
                   and p["text"][entity["offset"]-p["offset"]:entity["offset"]-p["offset"]+entity["length"]] == entity["text"]
                   for p in document["full_text"]):
            raise ValueError("A nested candidate differs from its exact source span.")
        additions.append(deepcopy(entity))
    result = sorted(deepcopy(reviewed) + additions, key=lambda entity: (entity["offset"], entity["length"]))
    return result, additions
