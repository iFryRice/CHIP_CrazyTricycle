"""Confidence-filtered UMLS additions that preserve a frozen CPU entity set."""

from collections import defaultdict
import copy
import csv
import gzip

from .entities import EntityExtractor, is_negated, phrase_key


def load_unambiguous_aliases(path, ontology):
    owners = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"term", "hpo_id"} <= set(reader.fieldnames or []):
            raise ValueError("UMLS export must contain term and hpo_id columns.")
        for row in reader:
            term, identifier = row["term"].strip(), row["hpo_id"]
            key = phrase_key(term)
            if identifier in ontology.allowed_ids and 5 <= len(term) <= 220 and 1 <= len(key) <= 24:
                if any(character.isalpha() for character in term):
                    owners[key].add(identifier)
    return {key: next(iter(values)) for key, values in owners.items() if len(values) == 1}


class UmlsAugmentedExtractor(EntityExtractor):
    """Train the existing context filter with extra unambiguous lexical matches.

    Labels and alias frequency statistics still come only from fit(documents).
    UMLS supplies candidate strings and concept IDs, never positive labels.
    """

    def __init__(self, ontology, extra_aliases):
        self.extra_aliases = dict(extra_aliases)
        self.added_keys = set()
        super().__init__(ontology, mode="learned")

    def _build_ontology_lexicon(self):
        super()._build_ontology_lexicon()
        self.added_keys = set(self.extra_aliases) - set(self.dictionary_ids)
        for key in sorted(self.added_keys):
            self.dictionary_ids[key].add(self.extra_aliases[key])
            self._insert(key)

    def score_additions(self, blind_document):
        if blind_document.get("entities") or blind_document.get("association"):
            raise ValueError("Prediction requires empty answer fields.")
        if self.model is None:
            raise RuntimeError("A fitted context filter with both classes is required.")
        candidates = [candidate for candidate in self._candidates(blind_document) if candidate.key in self.added_keys]
        if not candidates:
            return []
        scores = self.model.predict_proba(self.hasher.transform(self._features(c) for c in candidates))[:, 1]
        return [{"score": float(score), "entity": {
            "identifier": candidate.identifier, "type": "Phenotype", "offset": candidate.offset,
            "length": candidate.end - candidate.start, "text": candidate.text,
            "note": "NO" if is_negated(candidate.paragraph["text"], candidate.start, candidate.end) else None,
        }} for candidate, score in zip(candidates, scores, strict=True)]


def augment_entities(base_entities, scored_additions, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0, 1].")
    accepted = copy.deepcopy(base_entities)
    ordered = sorted(scored_additions, key=lambda item: (-item["entity"]["length"], -item["score"], item["entity"]["offset"], item["entity"]["identifier"]))
    for item in ordered:
        if item["score"] < threshold:
            continue
        entity = item["entity"]
        if any(entity["offset"] < other["offset"] + other["length"] and other["offset"] < entity["offset"] + entity["length"] for other in accepted):
            continue
        accepted.append(copy.deepcopy(entity))
    return sorted(accepted, key=lambda entity: (entity["offset"], entity["length"]))
