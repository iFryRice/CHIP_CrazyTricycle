"""Exact model-input identities for incremental concept/association inference."""

import hashlib
import json


def input_fingerprint(row):
    values = {key: row[key] for key in ["input_ids", "attention_mask", "token_type_ids"]}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def changed_examples(previous, current):
    def index(rows):
        result = {row["task_id"]: row for row in rows}
        if len(result) != len(rows): raise ValueError("Duplicate text example identity.")
        return result
    old, new = index(previous), index(current)
    if not set(new) <= set(old): raise ValueError("Deletion-only entity filtering created a new relation candidate.")
    changed = []
    for key, row in new.items():
        if any(row[field] != old[key][field] for field in ["pmc_id", "patient_id", "concept"]):
            raise ValueError("A task ID was reused for a different relation.")
        if input_fingerprint(row) != input_fingerprint(old[key]): changed.append(row)
    return changed
