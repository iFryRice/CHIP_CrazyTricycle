"""Evidence IDs must refer to the exact provided original-source fragments."""

import json
import unittest

from patientphex.llm_association_references import messages, parse_decision, prepare_prompt
from patientphex.llm_association import build_task
from tests.test_llm_association import fixture


class LlmAssociationReferenceTests(unittest.TestCase):
    def test_reference_ids_are_exhaustively_bounded_and_not_boolean(self):
        fragments = [{"text": "Patient 1 had seizures."}]
        self.assertEqual(parse_decision('{"decision":"supported","evidence_ids":[0]}', fragments)["evidence_ids"], [0])
        for identifiers in [[1], [-1], [True], ["0"], [0, 0], []]:
            with self.assertRaises(ValueError):
                parse_decision(json.dumps({"decision": "other_or_general", "evidence_ids": identifiers}), fragments)
        self.assertEqual(parse_decision('{"decision":"unclear","evidence_ids":[]}', fragments)["decision"], "unclear")

    def test_prompt_references_resolve_to_unmodified_original_excerpts(self):
        document, entity, ontology = fixture()
        task = build_task(document, [entity], "P1", entity["identifier"], 0.7, ontology)
        payload = json.loads(messages(task, task["fragments"])[1]["content"])
        for index, excerpt in enumerate(payload["original_source_excerpts"]):
            self.assertEqual(excerpt["excerpt_id"], index)
            self.assertEqual({key: value for key, value in excerpt.items() if key != "excerpt_id"}, task["fragments"][index])
        self.assertNotIn("baseline_score", payload)


if __name__ == "__main__":
    unittest.main()
