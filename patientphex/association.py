"""CPU patient--phenotype association using document-local mention evidence."""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


_GENERIC_MENTIONS = {
    "a", "an", "the", "he", "she", "his", "her", "they", "their", "we",
    "him", "patient", "patients", "the patient", "this patient", "individual",
    "woman", "man", "girl", "boy", "child", "infant", "female", "male",
    "the female", "the male", "the infant", "in this case", "case", "older",
    "younger", "daughter", "son",
}
_FAMILY_ROLE = re.compile(
    r"\b(?:grandfather|grandmother|father|mother|sister|brother|aunt|uncle|proband)\b",
    re.I,
)
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_STOP_WORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for", "is",
    "was", "were", "are", "with", "as", "by", "from", "that", "which", "be",
    "had", "has", "have", "been", "it", "this", "these", "also", "not",
}
_CLINICAL = re.compile(r"case|clinical|patient|proband|phenotyp|individual", re.I)
_GENERIC = re.compile(r"discuss|introduct|background|conclu|literature|review", re.I)
_SHARED = re.compile(r"\b(?:both|all (?:the |affected )?(?:patients|siblings|individuals)|shared)\b", re.I)


def _concepts(entity: dict[str, Any]) -> list[str]:
    """Expand a compound identifier, preserving original unmapped text."""
    if str(entity.get("note") or "").upper() == "NO":
        return []
    result = []
    for value in str(entity.get("identifier", "")).split(";"):
        value = value.strip()
        if value == "-1":
            value = entity.get("text", "")
        if value and value not in result:
            result.append(value)
    return result


def _gold_concepts(values: list[str]) -> set[str]:
    result = set()
    for value in values:
        if re.fullmatch(r"HP:\d+(?:\s*;\s*HP:\d+)+", value):
            result.update(part.strip() for part in value.split(";"))
        else:
            result.add(value)
    return result


@dataclass(frozen=True)
class _Marker:
    start: int
    end: int
    patient: str
    paragraph: int
    sentence: int
    explicit: bool


class _Document:
    """A read-only view of original offsets and paragraph/mention evidence."""

    def __init__(self, document: dict[str, Any]):
        self.patients = document.get("patient", [])
        self.paragraphs = sorted(document.get("full_text", []), key=lambda p: int(p["offset"]))
        self.starts = [int(p["offset"]) for p in self.paragraphs]
        self.sentence_starts: list[list[int]] = []
        self.blocks: list[int] = []
        self.headings: list[str] = []
        heading = ""
        block = 0
        for paragraph in self.paragraphs:
            text = paragraph.get("text", "")
            if "title" in paragraph.get("type", "").lower():
                heading = text
                block += 1
            self.headings.append(heading)
            self.blocks.append(block)
            self.sentence_starts.append([0] + [m.end() for m in re.finditer(r"[.!?](?:[\"')\]]*)\s+(?=[A-Z])", text)])
        markers: list[_Marker] = []
        alias_owners: dict[str, set[str]] = defaultdict(set)
        for patient in self.patients:
            patient_id = patient["patient_id"]
            for mention in patient.get("mention", []):
                start = int(mention["offset"])
                paragraph, sentence = self.location(start)
                markers.append(_Marker(start, start + int(mention.get("length", len(mention.get("text", "")))), patient_id, paragraph, sentence, True))
                alias = mention.get("text", "").strip(" ,:;.").lower()
                if len(alias) >= 3 and alias not in _GENERIC_MENTIONS and not alias.isdigit():
                    alias_owners[alias].add(patient_id)
                roles = _FAMILY_ROLE.findall(alias)
                # The final role identifies 'the proband\'s mother', not proband.
                if roles:
                    alias_owners[roles[-1].lower()].add(patient_id)
            if patient_id.startswith("O") and len(patient_id) >= 4:
                alias_owners[patient_id[1:].lower()].add(patient_id)
        for alias, owners in sorted(alias_owners.items()):
            if len(owners) != 1:
                continue
            patient_id = next(iter(owners))
            expression = re.compile(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", re.I)
            for paragraph_id, paragraph in enumerate(self.paragraphs):
                for match in expression.finditer(paragraph.get("text", "")):
                    start = int(paragraph["offset"]) + match.start()
                    _, sentence = self.location(start)
                    markers.append(_Marker(start, start + len(match.group()), patient_id, paragraph_id, sentence, False))
        # Preserve explicit markers when aliases identify the same span.
        unique: dict[tuple[int, int, str], _Marker] = {}
        for marker in markers:
            key = (marker.start, marker.end, marker.patient)
            unique.setdefault(key, marker)
        self.markers = sorted(unique.values(), key=lambda m: (m.start, m.end, m.patient))
        self.by_patient = {p["patient_id"]: [m for m in self.markers if m.patient == p["patient_id"]] for p in self.patients}
        self.by_paragraph: dict[int, list[_Marker]] = defaultdict(list)
        for marker in self.markers:
            self.by_paragraph[marker.paragraph].append(marker)

    def location(self, offset: int) -> tuple[int, int]:
        paragraph = bisect_right(self.starts, offset) - 1
        if paragraph < 0:
            return -1, -1
        source = self.paragraphs[paragraph]
        if offset >= int(source["offset"]) + len(source.get("text", "")):
            return -1, -1
        relative = offset - int(source["offset"])
        return paragraph, bisect_right(self.sentence_starts[paragraph], relative) - 1

    def occurrence(self, entity: dict[str, Any], patient_id: str) -> dict[str, float]:
        offset = int(entity["offset"])
        paragraph_id, sentence_id = self.location(offset)
        if paragraph_id < 0:
            return {"outside_text": 1.0}
        paragraph = self.paragraphs[paragraph_id]
        text = paragraph.get("text", "")
        relative = offset - int(paragraph["offset"])
        boundaries = self.sentence_starts[paragraph_id]
        start = boundaries[sentence_id]
        end = boundaries[sentence_id + 1] if sentence_id + 1 < len(boundaries) else len(text)
        sentence = text[start:end]
        local_markers = self.by_paragraph[paragraph_id]
        sentence_markers = [m for m in local_markers if m.sentence == sentence_id]
        own_markers = self.by_patient[patient_id]
        other_markers = [m for m in self.markers if m.patient != patient_id]
        distance = min((abs(offset - m.start) for m in own_markers), default=100000)
        other_distance = min((abs(offset - m.start) for m in other_markers), default=100000)
        preceding = [m for m in self.markers if m.start <= offset]
        preceding_local = [m for m in local_markers if m.start <= offset]
        previous = preceding[-1] if preceding else None
        local_previous = preceding_local[-1] if preceding_local else None
        nearest = min(self.markers, key=lambda m: (abs(offset - m.start), m.start), default=None)
        own_sentence = any(m.patient == patient_id for m in sentence_markers)
        other_sentence = any(m.patient != patient_id for m in sentence_markers)
        own_paragraph = any(m.patient == patient_id for m in local_markers)
        other_paragraph = any(m.patient != patient_id for m in local_markers)
        previous_own = previous is not None and previous.patient == patient_id
        previous_same_block = previous is not None and previous.paragraph >= 0 and self.blocks[previous.paragraph] == self.blocks[paragraph_id]
        section = paragraph.get("section_type", "UNKNOWN").upper()
        heading = self.headings[paragraph_id]
        clinical = section == "CASE" or bool(_CLINICAL.search(heading))
        generic = section in {"INTRO", "DISCUSS", "CONCL"} or bool(_GENERIC.search(heading))
        local_owner = local_previous.patient if local_previous else (nearest.patient if nearest and nearest.paragraph == paragraph_id else None)
        block_owner = previous.patient if previous_same_block else None
        features = {
            "section=" + section: 1.0,
            "type=" + paragraph.get("type", "unknown"): 1.0,
            "clinical": float(clinical),
            "generic": float(generic),
            "same_sentence": float(own_sentence),
            "other_sentence": float(other_sentence),
            "sole_sentence": float(own_sentence and not other_sentence),
            "same_paragraph": float(own_paragraph),
            "other_paragraph": float(other_paragraph),
            "sole_paragraph": float(own_paragraph and not other_paragraph),
            "nearest": float(nearest is not None and nearest.patient == patient_id),
            "preceding": float(previous_own),
            "preceding_block": float(previous_own and previous_same_block),
            "local_owner": float(local_owner == patient_id),
            "block_owner": float(block_owner == patient_id),
            "clinical_owner": float(clinical and (local_owner == patient_id or block_owner == patient_id)),
            "distance": 1.0 / (1.0 + distance / 200.0),
            "closer_than_others": float(distance <= other_distance),
            "near_100": float(distance < 100),
            "near_500": float(distance < 500),
            "near_2000": float(distance < 2000),
            "before_first": float(bool(own_markers) and offset < own_markers[0].start),
            "shared_sentence": float(bool(_SHARED.search(sentence)) and own_sentence),
            "relative_position": paragraph_id / max(len(self.paragraphs), 1),
        }
        for threshold in (0, 1, 3, 8):
            features[f"paragraph_distance_{threshold}"] = float(any(abs(paragraph_id - m.paragraph) <= threshold for m in own_markers if m.paragraph >= 0))
        # Lexical features are tied to local ownership rather than phenotype IDs.
        # This prevents learning document identities or memorizing disease labels.
        words = set(word.lower() for word in _WORD.findall(sentence)) - _STOP_WORDS
        for word in words:
            if len(word) >= 3:
                features["context=" + word] = 1.0
                if local_owner == patient_id or (not local_markers and block_owner == patient_id):
                    features["owned_context=" + word] = 1.0
        for word in set(word.lower() for word in _WORD.findall(heading)) - _STOP_WORDS:
            features["heading=" + word] = 1.0
        return features

    def samples(self, entities: list[dict[str, Any]]) -> list[tuple[str, str, dict[str, float]]]:
        concepts: dict[str, list[dict[str, Any]]] = {}
        for entity in sorted(entities, key=lambda e: (int(e["offset"]), str(e.get("identifier", "")))):
            for concept in _concepts(entity):
                concepts.setdefault(concept, []).append(entity)
        samples = []
        for patient in self.patients:
            patient_id = patient["patient_id"]
            for concept, occurrences in concepts.items():
                aggregate: dict[str, float] = {
                    "singleton": float(len(self.patients) == 1),
                    "patient_count": math.log1p(len(self.patients)),
                    "occurrence_count": math.log1p(len(occurrences)),
                    "marker_count": math.log1p(len(self.by_patient[patient_id])),
                    "unmapped": float(not concept.startswith("HP:")),
                }
                sums: Counter[str] = Counter()
                for entity in occurrences:
                    evidence = self.occurrence(entity, patient_id)
                    for key, value in evidence.items():
                        aggregate["max:" + key] = max(aggregate.get("max:" + key, 0.0), value)
                        if "=" not in key:
                            sums[key] += value
                for key, value in sums.items():
                    aggregate["mean:" + key] = value / len(occurrences)
                if len(self.patients) == 1:
                    for key, value in tuple(aggregate.items()):
                        if not key.startswith(("max:context=", "max:owned_context=")):
                            aggregate["single:" + key] = value
                samples.append((patient_id, concept, aggregate))
        return samples


class AssociationModel:
    """Fit document-local patient/concept evidence; never consult prediction gold."""

    def __init__(self, mode: str = "learned", threshold: float = 0.5):
        if mode not in {"nearest", "learned"}:
            raise ValueError("Association mode must be 'nearest' or 'learned'.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Association threshold must lie in [0, 1].")
        self.mode = mode
        self.threshold = threshold
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier: LogisticRegression | None = None
        self.constant: float | None = None
        self.is_fitted = False

    def fit(self, documents: list[dict[str, Any]], predictions: list[list[dict[str, Any]]] | None = None) -> "AssociationModel":
        if predictions is not None and len(predictions) != len(documents):
            raise ValueError("Training predictions must align one-to-one with documents.")
        self.classifier = None
        self.constant = None
        if self.mode == "nearest":
            self.is_fitted = True
            return self
        features = []
        labels = []
        for index, document in enumerate(documents):
            gold = {item["patient_id"]: _gold_concepts(item.get("phenotype", [])) for item in document.get("association", [])}
            entities = predictions[index] if predictions is not None else document.get("entities", [])
            for patient_id, concept, sample in _Document(document).samples(entities):
                features.append(sample)
                labels.append(int(concept in gold.get(patient_id, set())))
        if not features:
            self.constant = 0.0
        elif len(set(labels)) < 2:
            self.constant = float(labels[0])
        else:
            matrix = self.vectorizer.fit_transform(features)
            # Recent scipy/sklearn versions preserve 64-bit sparse indices;
            # liblinear requires 32-bit indices for this small CPU dataset.
            matrix.indices = matrix.indices.astype(np.int32, copy=False)
            matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
            self.classifier = LogisticRegression(C=0.5, max_iter=500, solver="liblinear", random_state=2026)
            self.classifier.fit(matrix, labels)
        self.is_fitted = True
        return self

    @staticmethod
    def _nearest_decision(features: dict[str, float]) -> bool:
        singleton = bool(features["singleton"])
        if features.get("max:sole_sentence", 0):
            return True
        if features.get("max:shared_sentence", 0):
            return True
        if features.get("max:clinical_owner", 0):
            return True
        if features.get("max:local_owner", 0) and features.get("max:same_paragraph", 0):
            return True
        return singleton and bool(features.get("max:clinical", 0))

    def predict(self, document: dict[str, Any], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.mode == "learned" and not self.is_fitted:
            raise RuntimeError("Fit the learned association model before predicting.")
        output = [{"patient_id": patient["patient_id"], "phenotype": []} for patient in document.get("patient", [])]
        targets = {item["patient_id"]: item["phenotype"] for item in output}
        samples = _Document(document).samples(entities)
        if not samples:
            return output
        if self.mode == "nearest":
            accepted = [self._nearest_decision(features) for _, _, features in samples]
        elif self.constant is not None:
            accepted = [self.constant >= self.threshold] * len(samples)
        else:
            matrix = self.vectorizer.transform([features for _, _, features in samples])
            probabilities = self.classifier.predict_proba(matrix)[:, 1]
            accepted = probabilities >= self.threshold
        for (patient_id, concept, _), accept in zip(samples, accepted):
            if accept:
                targets[patient_id].append(concept)
        return output
