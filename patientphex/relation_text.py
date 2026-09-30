"""Blind, source-grounded text examples for supervised patient associations."""

from collections import Counter
import hashlib
import json

from .association import _concepts
from .llm_association import _containing, _fragment
from .patient_linking import _training_gold


MARKERS = ["[TARGET]", "[/TARGET]", "[OTHER]", "[/OTHER]", "[PHENOTYPE]", "[/PHENOTYPE]"]


def marked_text(fragment, patients, target_id, entity=None):
    source, events = fragment["text"], {}
    if any(marker in source for marker in MARKERS):
        raise ValueError("Source collides with model-only markers.")

    def add(span, opening, closing):
        start = span["offset"] - fragment["offset"]
        end = start + span["length"]
        if start < 0 or end > len(source):
            return
        if source[start:end] != span["text"]:
            raise ValueError("Marker does not match the original source.")
        events.setdefault(start, []).append((1, opening))
        events.setdefault(end, []).append((0, closing))

    for patient in patients:
        pair = MARKERS[:2] if patient["patient_id"] == target_id else MARKERS[2:4]
        for mention in patient["mention"]:
            add(mention, *pair)
    if entity is not None:
        add(entity, *MARKERS[4:])
    return "".join("".join(value for _, value in sorted(events.get(i, [])))
                   + (source[i] if i < len(source) else "") for i in range(len(source)+1))


def examples(document, entities, ontology):
    if document.get("entities") or document.get("association"):
        raise ValueError("Text construction requires blind source documents.")
    concepts = {}
    for entity in entities:
        for concept in _concepts(entity):
            concepts.setdefault(concept, []).append(entity)
    result = []
    for patient in document["patient"]:
        anchors = patient["mention"]
        if not anchors:
            raise ValueError("Patient has no source anchors.")
        for anchor in anchors:
            _containing(document, anchor)
        for concept, occurrences in sorted(concepts.items()):
            chosen = sorted(occurrences, key=lambda e: (min(abs(e["offset"]-a["offset"]) for a in anchors),
                                                       e["offset"], e["length"]))[:3]
            fragments = []
            for entity in chosen:
                _, paragraph = _containing(document, entity)
                fragment = _fragment(paragraph, entity["offset"], entity["length"], 250)
                fragment["model_text"] = marked_text(fragment, document["patient"], patient["patient_id"], entity)
                fragments.append(fragment)
            anchor = min(anchors, key=lambda a: (min(abs(a["offset"]-e["offset"]) for e in chosen), a["offset"]))
            _, paragraph = _containing(document, anchor)
            reference = _fragment(paragraph, anchor["offset"], anchor["length"], 160)
            reference["model_text"] = marked_text(reference, document["patient"], patient["patient_id"])
            identity = [str(document["pmc_id"]), patient["patient_id"], concept]
            result.append({"task_id": hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
                           "pmc_id": identity[0], "patient_id": identity[1], "concept": concept,
                           "header": "phenotype: " + ontology.terms.get(concept, {}).get("name", concept),
                           "reference": reference, "fragments": fragments,
                           "occurrences": len(occurrences)})
    return result


def supervise(rows, documents, allowed_ids, forbidden_ids):
    indexed = {str(d["pmc_id"]): d for d in documents}
    if set(indexed) != set(allowed_ids) or set(indexed) & set(forbidden_ids):
        raise ValueError("Supervision leaves the permitted training partition.")
    if any(row["pmc_id"] not in indexed for row in rows):
        raise ValueError("An example received labels outside its training partition.")
    gold = {identifier: _training_gold(d) for identifier, d in indexed.items()}
    counts = Counter((row["pmc_id"], row["patient_id"]) for row in rows)
    scale = len(rows)/max(1, len(counts))
    result = []
    for row in rows:
        positive = row["concept"] in gold[row["pmc_id"]][row["patient_id"]]
        result.append({**row, "label": int(positive),
                       "weight": scale/counts[row["pmc_id"], row["patient_id"]]*(1.0 if positive else 0.5)})
    return result


def configure_tokenizer(tokenizer):
    before = len(tokenizer)
    missing = sum(tokenizer.convert_tokens_to_ids(marker) == tokenizer.unk_token_id for marker in MARKERS)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKERS})
    if len(tokenizer) != before + missing:
        raise ValueError("Vocabulary expansion differs from the six declared markers.")
    if any(tokenizer(marker, add_special_tokens=False)["input_ids"] != [tokenizer.convert_tokens_to_ids(marker)]
           for marker in MARKERS):
        raise ValueError("A declared marker is not encoded as one token.")


def centered_tokens(tokenizer, text, budget, focus):
    values = tokenizer(text, add_special_tokens=False)["input_ids"]
    opening, closing = [tokenizer.convert_tokens_to_ids(marker) for marker in focus]
    first = values.index(opening)
    last = values.index(closing, first+1)
    if last-first+1 > budget:
        raise ValueError("A focus span cannot fit its declared token budget.")
    start = min(max(0, (first+last+1-budget)//2), max(0, len(values)-budget))
    end = min(len(values), start+budget)
    if not start <= first <= last < end:
        raise ValueError("Token cropping removed the focus span.")
    return values[start:end], len(values)-(end-start)


def encode(tokenizer, row):
    header = tokenizer(row["header"], add_special_tokens=False)["input_ids"]
    cropped = max(0, len(header)-32)
    reference, count = centered_tokens(tokenizer, row["reference"]["model_text"], 64, MARKERS[:2])
    values = [tokenizer.cls_token_id, *header[:32], tokenizer.sep_token_id, *reference, tokenizer.sep_token_id]
    segments = [0]*len(values)
    cropped += count
    for fragment in row["fragments"]:
        snippet, count = centered_tokens(tokenizer, fragment["model_text"], 92, MARKERS[4:])
        values.extend([*snippet, tokenizer.sep_token_id])
        segments.extend([1]*(len(snippet)+1))
        cropped += count
    if len(values) > 384:
        raise ValueError("The fixed relation representation exceeds 384 tokens.")
    return {**row, "input_ids": values, "attention_mask": [1]*len(values), "token_type_ids": segments,
            "cropped_context_tokens": cropped}


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        import torch
        inputs = self.tokenizer.pad([{key: row[key] for key in ["input_ids", "attention_mask", "token_type_ids"]}
                                     for row in rows], padding=True, return_tensors="pt")
        result = {"inputs": inputs, "rows": rows}
        if "label" in rows[0]:
            result["labels"] = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
            result["weights"] = torch.tensor([r["weight"] for r in rows], dtype=torch.float32)
        return result


def build_model(encoder):
    import torch.nn as nn

    class RelationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(encoder.config.hidden_size, 1)

        def forward(self, **inputs):
            hidden = self.encoder(**inputs).last_hidden_state[:, 0]
            return self.classifier(self.dropout(hidden)).squeeze(-1)

    return RelationModel()
