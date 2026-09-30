"""Patient tags must respect provided identities and preserve original text."""

import copy
import re
import unittest

from patientphex.llm_association import build_task
from patientphex.llm_patient_markers import mark_task
from tests.test_llm_association import fixture


class PatientMarkerTests(unittest.TestCase):
    def test_target_identity_is_not_assumed_to_be_the_first_patient(self):
        document, entity, ontology = fixture()
        task = build_task(document, [entity], "P2", entity["identifier"], 0.7, ontology)
        before = copy.deepcopy(task)
        marked = mark_task(task, document["patient"])
        self.assertEqual(task, before)
        self.assertEqual([p["role_for_this_question"] for p in marked["patient_references"]], ["OTHER", "TARGET"])
        fragment = marked["fragments"][0]
        self.assertIn("[TARGET:P2]Patient 2[/TARGET:P2]", fragment["patient_annotated_text"])
        self.assertIn("[OTHER:P1]Patient 1[/OTHER:P1]", fragment["patient_annotated_text"])
        self.assertEqual(re.sub(r"\[/?(?:TARGET|OTHER):[^]]+\]", "", fragment["patient_annotated_text"]), fragment["text"])
        self.assertEqual(fragment["text"], task["fragments"][0]["text"])
        self.assertEqual(fragment["offset"], task["fragments"][0]["offset"])

    def test_wrong_anchor_text_and_missing_target_are_rejected(self):
        document, entity, ontology = fixture()
        task = build_task(document, [entity], "P2", entity["identifier"], 0.7, ontology)
        changed = copy.deepcopy(document["patient"])
        changed[0]["mention"][0]["text"] = "Patient X"
        with self.assertRaisesRegex(ValueError, "original source"):
            mark_task(task, changed)
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            mark_task(task, document["patient"][:1])


if __name__ == "__main__":
    unittest.main()
