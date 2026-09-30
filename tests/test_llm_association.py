"""Association review must preserve source evidence and uncertain predictions."""

import copy
import json
from types import SimpleNamespace
import unittest

from patientphex.llm_association import build_task, messages, parse_decision, prepare_prompt, should_veto


def fixture():
    text = "Patient 1 had seizures. Patient 2 had short stature."
    patients = [{"patient_id": f"P{number}", "mention": [{"offset": 100 + text.index(f"Patient {number}"),
                 "length": 9, "text": f"Patient {number}"}]} for number in (1, 2)]
    document = {"pmc_id": "test", "patient": patients, "entities": [], "association": [],
                "full_text": [{"offset": 100, "text": text, "section_type": "CASE", "type": "paragraph"}]}
    entity = {"offset": 114, "length": 8, "text": "seizures", "identifier": "HP:0001250", "note": None}
    return document, entity, SimpleNamespace(terms={"HP:0001250": {"name": "Seizure"}})


class LlmAssociationTests(unittest.TestCase):
    def test_tasks_use_blind_source_and_keep_other_patient_references(self):
        document, entity, ontology = fixture()
        before = copy.deepcopy(document)
        task = build_task(document, [entity], "P2", entity["identifier"], 0.7, ontology)
        payload = json.loads(messages(task, task["fragments"])[1]["content"])
        self.assertEqual(len(payload["patient_references"]), 2)
        self.assertNotIn("baseline_score", payload)
        self.assertNotIn("association", payload)
        self.assertEqual(document, before)
        with self.assertRaisesRegex(ValueError, "blind"):
            build_task({**document, "association": [{"patient_id": "P2", "phenotype": []}]}, [entity], "P2", entity["identifier"], 0.7, ontology)
        with self.assertRaisesRegex(ValueError, "exact source"):
            build_task(document, [{**entity, "text": "Seizures"}], "P2", entity["identifier"], 0.7, ontology)

    def test_truncated_occurrence_coverage_cannot_trigger_complete_policy(self):
        document, entity, ontology = fixture()
        # Distinct source occurrences, with the closest three selected.
        document["full_text"].extend({"offset": 200 + i * 100, "text": "seizures"} for i in range(3))
        entities = [entity] + [{**entity, "offset": 200 + i * 100} for i in range(3)]
        task = build_task(document, entities, "P1", entity["identifier"], 0.7, ontology)
        self.assertFalse(task["all_occurrences_selected"])
        self.assertFalse(should_veto("other_or_general", False, "complete_only"))
        self.assertTrue(should_veto("other_or_general", False, "all_reviewed"))
        for decision in ["supported", "unclear"]:
            self.assertFalse(should_veto(decision, True, "all_reviewed"))

    def test_parser_requires_exact_quoted_evidence_for_definite_decisions(self):
        fragments = [{"text": "Patient 1 had seizures."}]
        valid = '{"decision":"other_or_general","evidence":["Patient 1 had seizures."]}'
        self.assertEqual(parse_decision(valid, fragments)["decision"], "other_or_general")
        for invalid in ['{"decision":"supported","evidence":[]}',
                        '{"decision":"other_or_general","evidence":["Patient 2 had seizures."]}',
                        '{"decision":"unclear","evidence":[],"extra":1}']:
            with self.assertRaises(ValueError):
                parse_decision(invalid, fragments)

    def test_budget_removal_marks_incomplete_and_keeps_final_phenotype(self):
        class Tokenizer:
            def apply_chat_template(self, items, **kwargs):
                self.assert_nonthinking = kwargs["enable_thinking"] is False
                return "".join(item["content"] for item in items)

            def __call__(self, text, **kwargs):
                return {"input_ids": list(text)}

        document, entity, ontology = fixture()
        task = build_task(document, [entity], "P1", entity["identifier"], 0.7, ontology)
        tokenizer = Tokenizer()
        original = prepare_prompt(tokenizer, task, 10000, 20)
        task["fragments"].append({"offset": 999, "text": "x" * 1000, "role": "previous_context"})
        bounded = prepare_prompt(tokenizer, task, original["input_tokens"] + 30, 20)
        self.assertFalse(bounded["complete_evidence"])
        self.assertEqual(bounded["dropped_fragments"], 1)
        self.assertTrue(tokenizer.assert_nonthinking)
        with self.assertRaisesRegex(ValueError, "last phenotype"):
            prepare_prompt(tokenizer, task, 20, 10)


if __name__ == "__main__":
    unittest.main()
