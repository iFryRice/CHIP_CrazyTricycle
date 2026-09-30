"""BIO support filtering must preserve source entities and real overlap."""

import copy
import unittest

from patientphex.bio_corroboration import revise_entities


class BioCorroborationTests(unittest.TestCase):
    def test_veto_only_unsupported_positive_acronyms(self):
        entities = [{"offset": 0, "length": 3, "text": "CNS", "note": None},
                    {"offset": 10, "length": 2, "text": "ID", "note": "NO"},
                    {"offset": 20, "length": 7, "text": "seizure", "note": None}]
        original = copy.deepcopy(entities)
        kept, audit = revise_entities(entities, [], [], veto_acronyms=True)
        self.assertEqual(kept, entities[1:])
        self.assertEqual(audit["removed"], [entities[0]])
        self.assertEqual(entities, original)

    def test_support_requires_sufficient_overlap_and_confidence(self):
        entity = {"offset": 10, "length": 4, "text": "SNHL", "note": None}
        support = {"offset": 11, "length": 3, "scores": {"positive": 0.7}}
        self.assertEqual(revise_entities([entity], [support], [], veto_acronyms=True)[0], [entity])
        self.assertEqual(revise_entities([entity], [{**support, "scores": {"positive": 0.4}}], [], veto_acronyms=True)[0], [])
        self.assertEqual(revise_entities([entity], [{**support, "offset": 14}], [], veto_acronyms=True)[0], [])


if __name__ == "__main__":
    unittest.main()
