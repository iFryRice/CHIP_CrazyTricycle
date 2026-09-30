"""Patient-balanced nonlinear association from document-local structure."""

from collections import Counter
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer

from .association import AssociationModel, _Document, _concepts
from .patient_linking import _check_provenance, _patient_ids, _training_gold


def structural_samples(document, entities):
    """Avoid lexical identities and keep ownership conjunctions within occurrences."""
    view = _Document(document)
    by_concept = {}
    for entity in entities:
        for concept in _concepts(entity):
            by_concept.setdefault(concept, []).append(entity)
    result = []
    for patient, concept, features in view.samples(entities):
        filtered = {key: value for key, value in features.items()
                    if not any(token in key for token in ("context=", "owned_context=", "heading="))}
        occurrences = by_concept[concept]
        decisions, local_joint, explicit_owned, generic_owned = [], [], [], []
        for entity in occurrences:
            evidence = view.occurrence(entity, patient)
            decisions.append(AssociationModel._nearest_decision(
                {"singleton": len(view.patients) == 1, **{"max:" + key: value for key, value in evidence.items()}}))
            local_joint.append(bool(evidence.get("local_owner") and evidence.get("same_paragraph")))
            generic_owned.append(bool(evidence.get("local_owner") and evidence.get("generic")))
            paragraph, sentence = view.location(entity["offset"])
            explicit_owned.append(any(marker.explicit and marker.patient == patient and marker.sentence == sentence
                                      for marker in view.by_paragraph[paragraph]))
        for name, values in (("rule", decisions), ("local_joint", local_joint),
                             ("explicit_sentence", explicit_owned), ("generic_owned", generic_owned)):
            filtered["joint:any:" + name] = float(any(values))
            filtered["joint:mean:" + name] = sum(values) / len(values)
            filtered["joint:count:" + name] = math.log1p(sum(values))
        filtered["entity:min_length"] = min(entity["length"] for entity in occurrences)
        filtered["entity:max_length"] = max(entity["length"] for entity in occurrences)
        result.append((patient, concept, filtered))
    return result


class ContextTreeLinker:
    """Fit only explicitly permitted candidates; no patient or concept IDs as features."""

    def __init__(self, leaves=7):
        if leaves not in (7, 15):
            raise ValueError("Unsupported predeclared tree size.")
        self.leaves = leaves
        self.fitted = False

    def fit(self, documents, candidates, *, provenance, forbidden_document_ids):
        self.fitted = False
        identifiers = {str(d["pmc_id"]) for d in documents}
        forbidden = set(map(str, forbidden_document_ids))
        if len(identifiers) != len(documents) or identifiers & forbidden or set(candidates) != identifiers:
            raise ValueError("Training/candidate document coverage or partition violation.")
        self.provenance_ = _check_provenance(provenance, identifiers, forbidden)
        features, labels, owners = [], [], []
        for document in documents:
            gold = _training_gold(document)
            for patient, concept, row in structural_samples(document, candidates[str(document["pmc_id"])]):
                features.append(row)
                labels.append(int(concept in gold[patient]))
                owners.append((str(document["pmc_id"]), patient))
        counts = Counter(owners)
        scale = len(labels) / max(1, len(counts))
        weights = [scale / counts[owner] * (1.0 if label else 0.5) for owner, label in zip(owners, labels, strict=True)]
        self.vectorizer = DictVectorizer(sparse=False)
        self.constant = None
        if len(set(labels)) < 2:
            self.constant = float(labels[0]) if labels else 0.0
        else:
            matrix = self.vectorizer.fit_transform(features)
            self.classifier = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=150, max_leaf_nodes=self.leaves,
                min_samples_leaf=20, l2_regularization=5.0, early_stopping=False, random_state=20260929)
            self.classifier.fit(matrix, labels, sample_weight=np.asarray(weights))
        self.diagnostics_ = {"documents": len(documents), "pairs": len(labels), "positive_pairs": sum(labels),
                             "patients": len(counts), "features": len(self.vectorizer.feature_names_) if self.constant is None else 0,
                             "label_source": "Official training association sets only; absent links are negative surrogates.",
                             "candidate_scope": "Inner cross-fitted CPU entities; validation entities additionally include UMLS and neural semantic proposals."}
        self.fitted = True
        return self

    def predict_scores(self, document, entities):
        if not self.fitted:
            raise RuntimeError("Fit the association model first.")
        if document.get("entities") or document.get("association"):
            raise ValueError("Prediction requires a blind document.")
        _patient_ids(document)
        samples = structural_samples(document, entities)
        if not samples:
            return []
        probabilities = ([self.constant] * len(samples) if self.constant is not None else
                         self.classifier.predict_proba(self.vectorizer.transform([row for _, _, row in samples]))[:, 1])
        return [{"patient_id": patient, "concept": concept, "score": float(score)}
                for (patient, concept, _), score in zip(samples, probabilities, strict=True)]
