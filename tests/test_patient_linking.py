"""Leakage guards, patient balancing and association output contracts."""

import copy
import unittest

from patientphex.patient_linking import PatientLinkingModel


FIXED_PROVENANCE = {"kind": "fixed_knowledge_base", "uses_training_labels": False}


def document(identifier="train", *, two_patients=True):
    first = "Patient 1 has seizures and short stature."
    second = "Patient 2 has hearing loss, but no seizures."
    offset = len(first) + 10
    patients = [{"patient_id": "P1", "mention": [{"text": "Patient 1", "offset": 0, "length": 9}]}]
    associations = [{"patient_id": "P1", "phenotype": ["HP:0001250", "HP:0004322"]}]
    if two_patients:
        patients.append({"patient_id": "P2", "mention": [{"text": "Patient 2", "offset": offset, "length": 9}]})
        associations.append({"patient_id": "P2", "phenotype": ["HP:0000365"]})
    return {
        "pmc_id": identifier,
        "patient": patients,
        "full_text": [
            {"section_type": "CASE", "type": "paragraph", "offset": 0, "text": first},
            {"section_type": "CASE", "type": "paragraph", "offset": offset, "text": second},
        ],
        "entities": [{"identifier": "HP:9999999", "text": "unused gold entity", "offset": 0, "length": 18, "note": None}],
        "association": associations,
    }


def candidates(doc):
    first, second = doc["full_text"]
    return [
        {"identifier": "HP:0001250", "text": "seizures", "offset": first["text"].index("seizures"), "length": 8, "note": None},
        {"identifier": "HP:0004322", "text": "short stature", "offset": first["text"].index("short stature"), "length": 13, "note": None},
        {"identifier": "HP:0000365", "text": "hearing loss", "offset": second["offset"] + second["text"].index("hearing loss"), "length": 12, "note": None},
        {"identifier": "HP:0001250", "text": "seizures", "offset": second["offset"] + second["text"].index("seizures"), "length": 8, "note": "NO"},
    ]


def blind(doc):
    return {**doc, "entities": [], "association": []}


def fit(docs, proposed=None, **kwargs):
    mapping = proposed if proposed is not None else {doc["pmc_id"]: candidates(doc) for doc in docs}
    return PatientLinkingModel(**kwargs).fit(
        docs, mapping, provenance=FIXED_PROVENANCE, forbidden_document_ids=["validation", "B"],
    )


class PatientLinkingTests(unittest.TestCase):
    def test_requires_explicit_candidates_provenance_and_forbidden_ids(self):
        doc = document()
        with self.assertRaises(TypeError):
            PatientLinkingModel().fit([doc])
        with self.assertRaises(TypeError):
            PatientLinkingModel().fit([doc], {"train": candidates(doc)})
        with self.assertRaises(ValueError):
            fit([doc], {})
        with self.assertRaises(ValueError):
            fit([doc], {"wrong": candidates(doc)})
        with self.assertRaises(ValueError):
            fit([document("validation")])

    def test_fixed_source_rejects_training_labels(self):
        doc = document()
        for provenance in (
            {"kind": "fixed_knowledge_base"},
            {"kind": "fixed_knowledge_base", "uses_training_labels": True},
            {**FIXED_PROVENANCE, "training_document_ids": ["train"]},
        ):
            with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                PatientLinkingModel().fit([doc], {"train": candidates(doc)}, provenance=provenance, forbidden_document_ids=[])

    def test_oof_sources_reject_self_outer_validation_and_unknown_labels(self):
        docs = [document("one"), document("two")]
        proposed = {doc["pmc_id"]: candidates(doc) for doc in docs}
        valid = {"kind": "out_of_fold", "by_document": {
            "one": {"training_document_ids": ["two"], "label_source_document_ids": ["two"]},
            "two": {"training_document_ids": ["one"], "label_source_document_ids": ["one"]},
        }}
        model = PatientLinkingModel().fit(docs, proposed, provenance=valid, forbidden_document_ids=["validation"])
        self.assertEqual(model.training_diagnostics_["provenance_kind"], "out_of_fold")
        for source in ("one", "validation", "outside"):
            invalid = copy.deepcopy(valid)
            invalid["by_document"]["one"]["training_document_ids"].append(source)
            with self.subTest(source=source), self.assertRaises(ValueError):
                PatientLinkingModel().fit(docs, proposed, provenance=invalid, forbidden_document_ids=["validation"])
        for field in ("validation_document_ids", "selection_document_ids", "early_stopping_document_ids"):
            for complete_union in (False, True):
                invalid = copy.deepcopy(valid)
                invalid["by_document"]["one"][field] = ["validation"]
                if complete_union:
                    invalid["by_document"]["one"]["label_source_document_ids"].append("validation")
                with self.subTest(field=field, complete_union=complete_union), self.assertRaises(ValueError):
                    PatientLinkingModel().fit(docs, proposed, provenance=invalid, forbidden_document_ids=["validation"])
        missing = copy.deepcopy(valid)
        del missing["by_document"]["one"]["label_source_document_ids"]
        with self.assertRaises(ValueError):
            PatientLinkingModel().fit(docs, proposed, provenance=missing, forbidden_document_ids=["validation"])

    def test_coverage_uses_predictions_not_gold_entities_and_no_mutation(self):
        doc = document()
        proposed = candidates(doc)[:1]
        before = copy.deepcopy((doc, proposed, FIXED_PROVENANCE))
        model = fit([doc], {"train": proposed})
        info = model.training_diagnostics_
        self.assertEqual(info["candidate_pairs"], 2)
        self.assertEqual(info["covered_gold_relations"], 1)
        self.assertEqual(info["gold_relations"], 3)
        self.assertAlmostEqual(info["candidate_relation_recall_ceiling"], 1 / 3)
        self.assertEqual(info["missing_gold_relations"], 2)
        self.assertEqual((doc, proposed, FIXED_PROVENANCE), before)
        scores = model.predict_scores(blind(doc), proposed)
        self.assertEqual({item["concept"] for item in scores}, {"HP:0001250"})
        self.assertTrue(all(0 <= item["score"] <= 1 for item in scores))

    def test_balancing_is_per_patient_not_per_document_or_candidate_count(self):
        one = document("one", two_patients=False)
        two = document("two")
        proposed = {"one": candidates(one)[:1], "two": candidates(two)}
        model = fit([one, two], proposed)
        rows = model.training_diagnostics_["patients_detail"]
        self.assertEqual([row["candidate_pairs"] for row in rows], [1, 3, 3])
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertAlmostEqual(row["effective_weight_sum"], 7 / 3)
        downweighted = fit([one, two], proposed, negative_weight=0.25)
        altered = downweighted.training_diagnostics_["patients_detail"]
        self.assertAlmostEqual(altered[0]["effective_weight_sum"], 7 / 3)
        self.assertLess(altered[1]["effective_weight_sum"], rows[1]["effective_weight_sum"])
        self.assertIn("negative surrogates", downweighted.training_diagnostics_["negative_label_scope"])

    def test_no_exclusion_compounds_unmapped_duplicates_and_empty_patients(self):
        doc = document()
        entities = candidates(doc)
        entities[0]["identifier"] = "HP:0001250;HP:0004322"
        entities.append({**entities[0], "identifier": "-1", "text": "raw; Original text"})
        entities.append(copy.deepcopy(entities[0]))
        model = fit([doc], {"train": entities})
        result = model.predict(blind(doc), entities, threshold=0)
        expected = {"HP:0001250", "HP:0004322", "HP:0000365", "raw; Original text"}
        for row in result:
            self.assertEqual(set(row["phenotype"]), expected)
            self.assertEqual(len(row["phenotype"]), len(expected))
        self.assertEqual(model.predict(blind(doc), [entities[3]]), [
            {"patient_id": "P1", "phenotype": []}, {"patient_id": "P2", "phenotype": []},
        ])
        self.assertEqual(model.training_diagnostics_["excluded_no_entities"], 1)

    def test_no_singleton_shortcut_and_no_concept_identity_features(self):
        doc = document(two_patients=False)
        doc["association"][0]["phenotype"] = []
        model = fit([doc])
        self.assertEqual(model.predict(blind(doc), candidates(doc)), [{"patient_id": "P1", "phenotype": []}])
        mixed = fit([document()])
        names = mixed.vectorizer.get_feature_names_out()
        self.assertFalse(any("HP:" in name or "patient_id" in name or "pmc_id" in name for name in names))

    def test_predict_requires_blind_inputs_and_preserves_patient_ids(self):
        doc = document()
        model = fit([doc])
        with self.assertRaises(ValueError):
            model.predict_scores(doc, candidates(doc))
        target = blind(doc)
        target["patient"] = copy.deepcopy(target["patient"])
        target["patient"][1]["patient_id"] = "OII.2"
        result = model.predict(target, [])
        self.assertEqual([row["patient_id"] for row in result], ["P1", "OII.2"])

    def test_unfitted_empty_training_refit_and_invalid_coverage(self):
        doc = document()
        with self.assertRaises(RuntimeError):
            PatientLinkingModel().predict(blind(doc), candidates(doc))
        empty = fit([], {})
        self.assertIsNone(empty.training_diagnostics_["candidate_relation_recall_ceiling"])
        self.assertTrue(all(not row["phenotype"] for row in empty.predict(blind(doc), candidates(doc))))
        fitted = fit([doc])
        with self.assertRaises(ValueError):
            fitted.fit([doc], {}, provenance=FIXED_PROVENANCE, forbidden_document_ids=[])
        with self.assertRaises(RuntimeError):
            fitted.predict(blind(doc), candidates(doc))
        invalid = document()
        invalid["association"] = invalid["association"][:1]
        with self.assertRaises(ValueError):
            fit([invalid])


if __name__ == "__main__":
    unittest.main()
