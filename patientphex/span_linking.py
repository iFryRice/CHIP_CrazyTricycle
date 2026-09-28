"""Link predicted original-text spans to the competition HPO release.

Only ``fit`` reads annotations. Linking uses source text and saved span scores;
validation annotations never supply aliases, candidates, or tie-breaking counts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import unicodedata

from .entities import GAP, WORD, is_negated, normalize_token, phrase_key
from .ontology import Ontology


LABELS = ("positive", "NO")


def _strict_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def allowed_ids_digest(ontology: Ontology) -> str:
    return hashlib.sha256("\n".join(sorted(ontology.allowed_ids)).encode("utf-8")).hexdigest()


class SpanLinker:
    """Conservative exact/normalized lexicon linking with explicit ambiguity.

    Priority: fold-training exact alias, HPO exact name/synonym, fold-training
    normalized alias, HPO normalized name/synonym, UMLS exact crosswalk, UMLS
    normalized crosswalk. A training alias needs a unique candidate or a strict
    majority of document-level support. Unresolved ambiguity and missing terms
    produce ``-1``; distinct candidate IDs are never invented as a compound ID.
    """

    def __init__(self, ontology: Ontology, umls_terms_path: str | Path | None = None):
        if ontology.version != "2026-06-23" or "HP:0000118" in ontology.allowed_ids:
            raise ValueError("Linking requires HPO 2026-06-23 descendants excluding HP:0000118")
        self.ontology = ontology
        self.hpo_exact = defaultdict(set)
        self.hpo_normalized = defaultdict(set)
        self.umls_exact = defaultdict(set)
        self.umls_normalized = defaultdict(set)
        for identifier in sorted(ontology.allowed_ids):
            term = ontology.terms[identifier]
            for text in [term.get("name", ""), *term["exact_synonyms"]]:
                if _strict_key(text):
                    self.hpo_exact[_strict_key(text)].add(identifier)
                if phrase_key(text):
                    self.hpo_normalized[phrase_key(text)].add(identifier)
        self.resource_audit = {"hpo_version": ontology.version, "allowed_hpo_ids": len(ontology.allowed_ids),
                               "allowed_ids_sha256": allowed_ids_digest(ontology), "umls_rows": 0}
        if umls_terms_path is not None:
            self._load_umls(Path(umls_terms_path))
        self._knowledge_trie = None
        self.fit([])

    def _load_umls(self, path: Path) -> None:
        manifest_path = path.with_name(path.name + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("hpo_version") != self.ontology.version or manifest.get("allowed_ids_sha256") != allowed_ids_digest(self.ontology):
            raise ValueError("UMLS export was built for a different HPO version or allowed branch")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest.get("output_sha256"):
            raise ValueError("UMLS export hash does not match its manifest")
        rows = 0
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not reader.fieldnames or not {"term", "hpo_id", "cui", "sab"} <= set(reader.fieldnames):
                raise ValueError("UMLS export is missing required columns")
            for row in reader:
                identifier, text = row["hpo_id"], row["term"]
                if identifier not in self.ontology.allowed_ids:
                    raise ValueError(f"UMLS export contains a nonbranch HPO ID: {identifier}")
                if _strict_key(text):
                    self.umls_exact[_strict_key(text)].add(identifier)
                if phrase_key(text):
                    self.umls_normalized[phrase_key(text)].add(identifier)
                rows += 1
        if rows != manifest.get("exported_rows"):
            raise ValueError("UMLS export row count does not match its manifest")
        self.resource_audit.update({"umls_rows": rows, "umls_exact_phrases": len(self.umls_exact),
                                   "umls_normalized_phrases": len(self.umls_normalized),
                                   "umls_export_sha256": digest, "umls_export_path": str(path),
                                   "umls_source_version": manifest.get("umls_hpo_source_versions")})

    def fit(self, training_documents: Iterable[dict], *, forbidden_document_ids: Iterable[str] = ()) -> "SpanLinker":
        """Reset all learned aliases; optional held-out IDs enforce fold isolation."""
        self.training_exact = defaultdict(Counter)
        self.training_normalized = defaultdict(Counter)
        forbidden = {str(identifier) for identifier in forbidden_document_ids}
        seen = set()
        invalid = []
        for document in training_documents:
            doc_id = str(document["pmc_id"])
            if doc_id in seen or doc_id in forbidden:
                raise ValueError(f"Duplicate or forbidden training document: {doc_id}")
            seen.add(doc_id)
            exact_votes, normalized_votes = set(), set()
            for index, entity in enumerate(document.get("entities", [])):
                identifier = self.ontology.normalize_identifier(entity["identifier"])
                if identifier is None:
                    invalid.append({"pmc_id": doc_id, "entity_index": index, "identifier": entity["identifier"]})
                    continue
                exact, normalized = _strict_key(entity["text"]), phrase_key(entity["text"])
                if exact:
                    exact_votes.add((exact, identifier))
                if normalized:
                    normalized_votes.add((normalized, identifier))
            for key, identifier in exact_votes:
                self.training_exact[key][identifier] += 1
            for key, identifier in normalized_votes:
                self.training_normalized[key][identifier] += 1
        self.training_document_ids = seen
        self.fit_audit = {"training_document_ids": sorted(seen), "documents": len(seen),
                          "exact_aliases": len(self.training_exact), "normalized_aliases": len(self.training_normalized),
                          "invalid_identifier_annotations": invalid}
        return self

    def resolve(self, text: str) -> dict:
        """Resolve one complete mention without changing its character extent."""
        exact, normalized = _strict_key(text), phrase_key(text)
        tiers = (
            ("training_exact", self.training_exact.get(exact)),
            ("hpo_exact", self.hpo_exact.get(exact)),
            ("training_normalized", self.training_normalized.get(normalized)),
            ("hpo_normalized", self.hpo_normalized.get(normalized)),
            ("umls_exact", self.umls_exact.get(exact)),
            ("umls_normalized", self.umls_normalized.get(normalized)),
        )
        for source, candidates in tiers:
            if not candidates:
                continue
            identifiers = sorted(candidates)
            if len(identifiers) == 1:
                return {"identifier": identifiers[0], "source": source, "candidates": identifiers, "ambiguous": False}
            if isinstance(candidates, Counter):
                top, count = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[0]
                if count * 2 > sum(candidates.values()):
                    return {"identifier": top, "source": source + "_majority", "candidates": identifiers,
                            "document_support": dict(sorted(candidates.items())), "ambiguous": False}
            return {"identifier": "-1", "source": source, "candidates": identifiers, "ambiguous": True}
        return {"identifier": "-1", "source": "no_candidate", "candidates": [], "ambiguous": False}

    def knowledge_spans(self, document: dict) -> dict:
        """Match only fixed HPO/UMLS knowledge, independently of ``fit`` aliases.

        Nested and overlapping spans are retained. One-character tokens are
        excluded; matches of at most three characters must preserve uppercase
        spelling. Negation is the existing explicit local rule, not a gold note.
        Scores of one denote deterministic candidate presence, not confidence.
        """
        if self._knowledge_trie is None:
            trie = {}
            for key in sorted(set(self.hpo_normalized) | set(self.umls_normalized)):
                node = trie
                for token in key:
                    node = node.setdefault(token, {})
                node[None] = True
            self._knowledge_trie = trie
        matches = {}
        for paragraph in document["full_text"]:
            text = paragraph["text"]
            tokens = list(WORD.finditer(text))
            normalized = [normalize_token(token[0]) for token in tokens]
            for index, token in enumerate(tokens):
                node = self._knowledge_trie
                for end_index in range(index, len(tokens)):
                    if end_index > index and not GAP.fullmatch(text[tokens[end_index - 1].end():tokens[end_index].start()]):
                        break
                    node = node.get(normalized[end_index])
                    if node is None:
                        break
                    if None not in node:
                        continue
                    start, end = token.start(), tokens[end_index].end()
                    mention = text[start:end]
                    if len(mention) == 1 or len(mention) <= 3 and not mention.isupper():
                        continue
                    offset = paragraph["offset"] + start
                    label = "NO" if is_negated(text, start, end) else "positive"
                    span = matches.setdefault((offset, end - start), {
                        "offset": offset, "length": end - start, "text": mention, "labels": [], "scores": {},
                    })
                    if label not in span["scores"]:
                        span["labels"].append(label)
                        span["scores"][label] = 1.0
        return {"pmc_id": document["pmc_id"], "spans": [matches[key] for key in sorted(matches)]}

    def link_document(self, document: dict, span_record: dict, *, thresholds: dict[str, float] | None = None) -> tuple[list[dict], dict]:
        """Return competition entities plus diagnostics for the saved span format.

        Threshold comparison is inclusive. If both labels pass, the greater raw
        model score wins (positive wins an exact tie), and the conflict is logged.
        This is an explicit arbitration rule, not a claim of score calibration.
        Source ``entities`` and ``association`` are never inspected here.
        """
        thresholds = dict({"positive": 0.5, "NO": 0.5} if thresholds is None else thresholds)
        if set(thresholds) != set(LABELS) or any(type(value) not in (int, float) or not math.isfinite(value)
                                               or not 0 <= value <= 1 for value in thresholds.values()):
            raise ValueError("Supply finite positive and NO thresholds in [0, 1]")
        if str(document["pmc_id"]) != str(span_record["pmc_id"]):
            raise ValueError("Span document ID differs from its original source")
        entities, details, conflicts = [], [], []
        seen = set()
        below_threshold = 0
        for index, span in enumerate(span_record["spans"]):
            offset, length, text = span["offset"], span["length"], span["text"]
            if (type(offset) is not int or offset < 0 or type(length) is not int or length <= 0
                    or not isinstance(text, str) or not text.strip()):
                raise ValueError(f"Malformed span at index {index}")
            if (offset, length) in seen:
                raise ValueError("Merge repeated spans and their label scores before linking")
            seen.add((offset, length))
            if not any(p["offset"] <= offset and offset + length <= p["offset"] + len(p["text"])
                       and p["text"][offset - p["offset"]:offset - p["offset"] + length] == text
                       for p in document["full_text"]):
                raise ValueError(f"Span does not match unchanged original text at index {index}")
            labels, scores = span["labels"], span["scores"]
            if not labels or len(labels) != len(set(labels)) or set(labels) != set(scores) or not set(labels) <= set(LABELS):
                raise ValueError(f"Malformed span label scores at index {index}")
            if any(type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores.values()):
                raise ValueError(f"Nonfinite or out-of-range span score at index {index}")
            accepted = [label for label in labels if scores[label] >= thresholds[label]]
            if not accepted:
                below_threshold += 1
                continue
            label = max(accepted, key=lambda name: (scores[name], name == "positive"))
            if len(accepted) > 1:
                conflicts.append({"span_index": index, "offset": offset, "length": length,
                                  "scores": dict(scores), "selected_label": label})
            resolution = self.resolve(text)
            entities.append({"identifier": resolution["identifier"], "type": "Phenotype", "offset": offset,
                             "length": length, "text": text, "note": "NO" if label == "NO" else None})
            details.append({"span_index": index, "offset": offset, "length": length, **resolution})
        entities.sort(key=lambda entity: (entity["offset"], entity["length"], entity["identifier"]))
        return entities, {
            "pmc_id": str(document["pmc_id"]), "input_spans": len(span_record["spans"]), "output_entities": len(entities),
            "below_threshold": below_threshold, "unmapped": sum(entity["identifier"] == "-1" for entity in entities),
            "ambiguous": sum(detail["ambiguous"] for detail in details), "thresholds": thresholds,
            "label_conflict_policy": "Highest raw score among passing labels; positive on an exact tie",
            "label_conflicts": conflicts, "resolutions": details,
        }
