"""Protect candidate freezing and predeclared promotion for association ablations."""

import copy
import unittest

from scripts.evaluate_association_reweighting import check_partition, promotion_decision, replace_associations


class AssociationReweightingTests(unittest.TestCase):
    def test_rejects_overlapping_or_incomplete_fold(self):
        check_partition(["a"], ["b"], {"a", "b"})
        for train, validation in ((["a"], ["a", "b"]), (["a"], []), (["a", "a"], ["b"])):
            with self.subTest(train=train, validation=validation), self.assertRaises(ValueError):
                check_partition(train, validation, {"a", "b"})

    def test_predictions_are_blind_and_entity_outputs_are_preserved(self):
        entity = {"identifier": "HP:0001250", "text": "seizures", "offset": 0, "length": 8, "note": None}
        document = {"pmc_id": "a", "pmid": "1", "patient": [{"patient_id": "P1"}],
                    "entities": [entity], "association": [{"patient_id": "P1", "phenotype": ["HP:0001250"]}]}
        frozen = [{"pmc_id": "a", "pmid": "1", "entities": [entity], "association": []}]
        snapshot = copy.deepcopy((document, frozen))

        class Model:
            def predict(self, source, entities, threshold):
                if source["entities"] or source["association"]:
                    raise AssertionError("Gold reached prediction")
                return [{"patient_id": "P1", "phenotype": ["HP:0001250"]}]

        output = replace_associations([document], frozen, Model(), 0.3)
        self.assertEqual((document, frozen), snapshot)
        self.assertEqual(output[0]["entities"], frozen[0]["entities"])
        self.assertNotEqual(output[0]["association"], frozen[0]["association"])
        with self.assertRaises(ValueError):
            replace_associations([document], [], Model(), 0.3)

    def test_promotion_rejects_average_gain_with_regression_or_false_assignment_cost(self):
        base = {"score": 0.60, "mention": {"f1": 0.65}, "document": {"f1": 0.70},
                "association_micro": {"f1": 0.55, "precision": 0.62}, "association_macro": {"f1": 0.50}}
        new = copy.deepcopy(base)
        new.update(score=0.61, association_micro={"f1": 0.57, "precision": 0.60}, association_macro={"f1": 0.52})
        criteria = {"minimum_score_gain": 0.002, "minimum_nonworse_folds": 4, "maximum_micro_precision_loss": 0.05}
        before, after = {"reachable": 100, "reachable_but_missed": 30}, {"reachable": 100, "reachable_but_missed": 20}
        self.assertTrue(promotion_decision(base, new, [0.01] * 5, before, after, criteria)["passed"])
        self.assertFalse(promotion_decision(base, new, [0.1, -0.01, -0.01, -0.01, -0.01], before, after, criteria)["passed"])
        for key, value in (("precision", 0.55), ("f1", 0.54)):
            altered = copy.deepcopy(new)
            altered["association_micro"][key] = value
            self.assertFalse(promotion_decision(base, altered, [0.01] * 5, before, after, criteria)["passed"])
        changed_entities = copy.deepcopy(new)
        changed_entities["mention"]["f1"] = 0.70
        self.assertFalse(promotion_decision(base, changed_entities, [0.01] * 5, before, after, criteria)["passed"])


if __name__ == "__main__":
    unittest.main()
