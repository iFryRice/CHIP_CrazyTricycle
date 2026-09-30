"""Semantic review uses original source and protects unambiguous ontology terms."""

import copy
from types import SimpleNamespace
import unittest

from patientphex.llm_concept_review import build_task, eligible


class ConceptReviewTests(unittest.TestCase):
    def fixture(self):
        text = "Noonan syndrome (NS) was diagnosed. NS features varied."
        document = {"pmc_id": "a", "full_text": [{"offset": 100, "text": text}], "entities": [], "association": []}
        entity = {"offset": 136, "length": 2, "text": "NS", "identifier": "HP:1", "note": None, "type": "Phenotype"}
        ontology = SimpleNamespace(allowed_ids={"HP:1"}, terms={"HP:1": {"name": "Nephrotic syndrome", "exact_synonyms": []}})
        return document, entity, ontology

    def test_local_abbreviation_definition_and_original_span_are_preserved(self):
        document, entity, ontology = self.fixture()
        before = copy.deepcopy(document)
        task = build_task(document, entity, 0, ontology, {})
        self.assertEqual(task["document_definitions"][0]["long"], "Noonan syndrome")
        self.assertEqual(task["entity"], entity)
        self.assertEqual(document, before)
        for fragment in task["fragments"]:
            offset = fragment["offset"]-100
            self.assertEqual(fragment["text"], document["full_text"][0]["text"][offset:offset+len(fragment["text"])])
        with self.assertRaisesRegex(ValueError, "blind"):
            build_task({**document, "entities": [entity]}, entity, 0, ontology, {})

    def test_no_negated_unmapped_compound_or_exact_unique_hpo_term_is_reviewed(self):
        _, entity, ontology = self.fixture()
        linker = SimpleNamespace(hpo_exact={})
        self.assertTrue(eligible(entity, ontology, linker))
        self.assertFalse(eligible({**entity, "note": "NO"}, ontology, linker))
        self.assertFalse(eligible({**entity, "identifier": "-1"}, ontology, linker))
        self.assertFalse(eligible({**entity, "identifier": "HP:1;HP:2"}, ontology, linker))
        self.assertFalse(eligible(entity, ontology, SimpleNamespace(hpo_exact={"ns": {"HP:1"}})))


if __name__ == "__main__":
    unittest.main()
