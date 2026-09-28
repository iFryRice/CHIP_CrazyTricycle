"""Patient-balanced association learning from explicitly sourced candidates."""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .association import _Document, _concepts, _gold_concepts


def _identifier_set(values: Collection[str], name: str) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise ValueError(f"{name} must be an explicit collection of document IDs.")
    return {str(value) for value in values}


def _patient_ids(document: dict[str, Any]) -> list[str]:
    identifiers = [patient["patient_id"] for patient in document.get("patient", [])]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate patient IDs in {document.get('pmc_id')}.")
    return identifiers


def _training_gold(document: dict[str, Any]) -> dict[str, set[str]]:
    patients = _patient_ids(document)
    associations = document.get("association", [])
    identifiers = [item["patient_id"] for item in associations]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(patients):
        raise ValueError(f"Training association coverage mismatch in {document['pmc_id']}.")
    return {item["patient_id"]: _gold_concepts(item.get("phenotype", [])) for item in associations}


def _check_provenance(
    provenance: Mapping[str, Any], training_ids: set[str], forbidden_ids: set[str]
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("Explicit candidate provenance is required.")
    kind = provenance.get("kind")
    if kind == "fixed_knowledge_base":
        if provenance.get("uses_training_labels") is not False:
            raise ValueError("Fixed knowledge-base candidates must declare uses_training_labels=false.")
        source_ids = _identifier_set(provenance.get("training_document_ids", []), "training_document_ids")
        if source_ids:
            raise ValueError("Fixed knowledge-base candidate generation cannot use training document labels.")
    elif kind == "out_of_fold":
        by_document = provenance.get("by_document")
        if not isinstance(by_document, Mapping):
            raise ValueError("Out-of-fold provenance requires by_document metadata.")
        normalized = {str(key): value for key, value in by_document.items()}
        if len(normalized) != len(by_document) or set(normalized) != training_ids:
            raise ValueError("Out-of-fold provenance must cover exactly the training documents.")
        for identifier, metadata in normalized.items():
            if not isinstance(metadata, Mapping) or not {"training_document_ids", "label_source_document_ids"} <= set(metadata):
                raise ValueError(f"Missing complete upstream label-source declaration for {identifier}.")
            source_ids = _identifier_set(metadata["label_source_document_ids"], "label_source_document_ids")
            for field in ("training_document_ids", "validation_document_ids", "selection_document_ids",
                          "calibration_document_ids", "early_stopping_document_ids", "dev_document_ids", "test_document_ids"):
                declared = _identifier_set(metadata.get(field, []), field)
                if not declared <= source_ids:
                    raise ValueError(f"Incomplete upstream label-source declaration for {identifier}: {field}.")
            if identifier in source_ids:
                raise ValueError(f"Candidate leakage: {identifier} supplied upstream training/selection labels.")
            if source_ids & forbidden_ids or not source_ids <= training_ids:
                raise ValueError(f"Candidate leakage: upstream labels for {identifier} leave the allowed training set.")
        for field in ("training_document_ids", "label_source_document_ids", "validation_document_ids", "selection_document_ids",
                      "calibration_document_ids", "early_stopping_document_ids", "dev_document_ids", "test_document_ids"):
            source_ids = _identifier_set(provenance.get(field, []), field)
            if source_ids & forbidden_ids or not source_ids <= training_ids:
                raise ValueError("Candidate leakage in global upstream training document IDs.")
    else:
        raise ValueError("Candidate provenance kind must be fixed_knowledge_base or out_of_fold.")
    return copy.deepcopy(dict(provenance))


class PatientLinkingModel:
    """Score patient/concept bags without treating retrieved evidence as labels.

    Candidate generation is a separate step. Labels come only from the supplied
    training documents' association sets. Missing links are negative surrogates,
    not independently verified negative annotations.
    """

    def __init__(self, threshold: float = 0.5, C: float = 0.5, negative_weight: float = 1.0):
        self._check_threshold(threshold)
        if not math.isfinite(C) or C <= 0:
            raise ValueError("C must be positive and finite.")
        if not math.isfinite(negative_weight) or not 0 < negative_weight <= 1:
            raise ValueError("negative_weight must lie in (0, 1].")
        self.threshold = float(threshold)
        self.C = float(C)
        self.negative_weight = float(negative_weight)
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier: LogisticRegression | None = None
        self.constant: float | None = None
        self.is_fitted = False
        self.training_diagnostics_: dict[str, Any] = {}
        self.provenance_: dict[str, Any] = {}

    @staticmethod
    def _check_threshold(threshold: float) -> None:
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Association threshold must lie in [0, 1].")

    def fit(
        self,
        train_documents: Sequence[dict[str, Any]],
        candidates_by_pmc: Mapping[str, Sequence[dict[str, Any]]],
        *,
        provenance: Mapping[str, Any],
        forbidden_document_ids: Collection[str],
    ) -> "PatientLinkingModel":
        """Fit only explicit candidates, with document-level leakage guards."""
        self.is_fitted = False
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = None
        self.constant = None
        self.training_diagnostics_ = {}
        self.provenance_ = {}
        identifiers = [str(document["pmc_id"]) for document in train_documents]
        training_ids = set(identifiers)
        if len(training_ids) != len(identifiers):
            raise ValueError("Duplicate training document IDs.")
        forbidden_ids = _identifier_set(forbidden_document_ids, "forbidden_document_ids")
        if training_ids & forbidden_ids:
            raise ValueError("Training documents overlap forbidden validation/target documents.")
        if not isinstance(candidates_by_pmc, Mapping):
            raise ValueError("Candidates must be an explicit mapping keyed by pmc_id.")
        candidates = {str(key): value for key, value in candidates_by_pmc.items()}
        if len(candidates) != len(candidates_by_pmc) or set(candidates) != training_ids:
            raise ValueError("Candidates must cover exactly the training document IDs.")
        self.provenance_ = _check_provenance(provenance, training_ids, forbidden_ids)

        features: list[dict[str, float]] = []
        labels: list[int] = []
        owners: list[tuple[str, str]] = []
        patient_rows: list[dict[str, Any]] = []
        document_rows: list[dict[str, Any]] = []
        candidate_count = excluded_no = total_gold = covered_gold = 0
        for document in train_documents:
            identifier = str(document["pmc_id"])
            gold = _training_gold(document)
            entities = candidates[identifier]
            if not isinstance(entities, Sequence) or isinstance(entities, (str, bytes)):
                raise ValueError(f"Candidates for {identifier} must be an entity sequence.")
            concepts = {concept for entity in entities for concept in _concepts(entity)}
            no_count = sum(str(entity.get("note") or "").upper() == "NO" for entity in entities)
            candidate_count += len(entities)
            excluded_no += no_count
            samples = _Document(document).samples(entities)
            for patient_id, concept, sample in samples:
                features.append(sample)
                labels.append(int(concept in gold[patient_id]))
                owners.append((identifier, patient_id))
            for patient_id in _patient_ids(document):
                positives = len(gold[patient_id] & concepts)
                missing = sorted(gold[patient_id] - concepts)
                total_gold += len(gold[patient_id])
                covered_gold += positives
                patient_rows.append({
                    "pmc_id": identifier,
                    "patient_id": patient_id,
                    "candidate_pairs": len(concepts),
                    "positive_pairs": positives,
                    "negative_surrogate_pairs": len(concepts) - positives,
                    "gold_relations": len(gold[patient_id]),
                    "covered_gold_relations": positives,
                    "missing_gold_concepts": missing,
                })
            document_rows.append({
                "pmc_id": identifier,
                "candidate_entities": len(entities),
                "excluded_no_entities": no_count,
                "eligible_concepts": len(concepts),
            })

        counts = Counter(owners)
        # Each active patient has equal total weight before optional negative
        # downweighting. At the default, weights have mean one, keeping C stable.
        balance_scale = len(labels) / len(counts) if counts else 0.0
        weights = np.asarray([
            balance_scale / counts[owner] * (1.0 if label else self.negative_weight)
            for owner, label in zip(owners, labels, strict=True)
        ], dtype=np.float64)
        weight_sums: Counter[tuple[str, str]] = Counter()
        for owner, weight in zip(owners, weights, strict=True):
            weight_sums[owner] += float(weight)
        for row in patient_rows:
            owner = (row["pmc_id"], row["patient_id"])
            row["balanced_weight_before_negative_downweighting"] = balance_scale if counts[owner] else 0.0
            row["effective_weight_sum"] = weight_sums[owner]

        if not features:
            self.constant = 0.0
        elif len(set(labels)) == 1:
            self.constant = float(labels[0])
        else:
            matrix = self.vectorizer.fit_transform(features)
            matrix.indices = matrix.indices.astype(np.int32, copy=False)
            matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
            self.classifier = LogisticRegression(
                C=self.C, max_iter=500, solver="liblinear", random_state=20260928,
            )
            self.classifier.fit(matrix, labels, sample_weight=weights)

        self.training_diagnostics_ = {
            "documents": len(train_documents),
            "patients": len(patient_rows),
            "active_patients": len(counts),
            "candidate_entities": candidate_count,
            "excluded_no_entities": excluded_no,
            "candidate_pairs": len(labels),
            "positive_pairs": sum(labels),
            "negative_surrogate_pairs": len(labels) - sum(labels),
            "negative_weight": self.negative_weight,
            "effective_sample_weight_sum": float(weights.sum()),
            "gold_relations": total_gold,
            "covered_gold_relations": covered_gold,
            "candidate_relation_recall_ceiling": covered_gold / total_gold if total_gold else None,
            "missing_gold_relations": total_gold - covered_gold,
            "provenance_kind": self.provenance_["kind"],
            "candidate_distribution": (
                "Fixed knowledge-base candidates; not neural out-of-fold predictions."
                if self.provenance_["kind"] == "fixed_knowledge_base"
                else "Out-of-fold candidates with declared complete upstream training/validation/selection label-source IDs."
            ),
            "label_source": "Official association sets in train_documents only.",
            "negative_label_scope": "Absent associations are negative surrogates; annotation completeness is not assumed.",
            "evidence_scope": "Automatically derived document-local features; no human evidence-chain labels.",
            "coverage_scope": "Candidate reachability on training documents, not model accuracy or a validation score.",
            "patients_detail": patient_rows,
            "documents_detail": document_rows,
        }
        self.is_fitted = True
        return self

    def predict_scores(
        self, blind_document: dict[str, Any], entities: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return one probability per eligible patient/concept bag."""
        if not self.is_fitted:
            raise RuntimeError("Fit the patient linking model before predicting.")
        if blind_document.get("entities") or blind_document.get("association"):
            raise ValueError("Prediction requires a blind document with empty answer fields.")
        _patient_ids(blind_document)
        samples = _Document(blind_document).samples(entities)
        if not samples:
            return []
        if self.constant is not None:
            probabilities = [self.constant] * len(samples)
        else:
            matrix = self.vectorizer.transform([features for _, _, features in samples])
            probabilities = self.classifier.predict_proba(matrix)[:, 1]
        return [
            {"patient_id": patient_id, "concept": concept, "score": float(score)}
            for (patient_id, concept, _), score in zip(samples, probabilities, strict=True)
        ]

    def predict(
        self,
        blind_document: dict[str, Any],
        entities: Sequence[dict[str, Any]],
        *,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        cutoff = self.threshold if threshold is None else threshold
        self._check_threshold(cutoff)
        scores = self.predict_scores(blind_document, entities)
        output = [{"patient_id": patient_id, "phenotype": []} for patient_id in _patient_ids(blind_document)]
        targets = {item["patient_id"]: item["phenotype"] for item in output}
        for item in scores:
            if item["score"] >= cutoff:
                targets[item["patient_id"]].append(item["concept"])
        return output
