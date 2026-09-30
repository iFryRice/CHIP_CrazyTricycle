"""LLM support filtering must preserve negation, unmapped text and offsets."""

import copy
import unittest

from patientphex.llm_corroboration import filter_entities


def entity(text, offset, identifier="HP:0001250", note=None):
    return {"text": text, "offset": offset, "length": len(text), "identifier": identifier, "note": note}


class LlmCorroborationTests(unittest.TestCase):
    def test_all_scopes_preserve_negated_and_unmapped_entities_without_mutation(self):
        values = [entity("NS", 10), entity("NS", 30, note="NO"), entity("odd symptom", 50, identifier="-1")]
        original = copy.deepcopy(values)
        for scope in ["acronyms", "short_terms", "all_mapped"]:
            kept, audit = filter_entities(values, [], scope)
            self.assertEqual(kept, values[1:])
            self.assertEqual(audit["removed"], values[:1])
        self.assertEqual(values, original)

    def test_support_is_global_overlap_not_a_same_phrase_elsewhere(self):
        mention = entity("SNHL", 10)
        elsewhere = [{"offset": 40, "length": 4, "text": "SNHL"}]
        self.assertEqual(filter_entities([mention], elsewhere, "acronyms")[0], [])
        partial = [{"offset": 12, "length": 4, "text": "HLxx"}]
        self.assertEqual(filter_entities([mention], partial, "acronyms")[0], [mention])
        one_character = [{"offset": 10, "length": 1, "text": "S"}]
        self.assertEqual(filter_entities([mention], one_character, "acronyms")[0], [])

    def test_policy_scope_is_explicit(self):
        values = [entity("NS", 10), entity("seizure", 30), entity("short stature", 50)]
        self.assertEqual(filter_entities(values, [], "acronyms")[0], values[1:])
        self.assertEqual(filter_entities(values, [], "short_terms")[0], values[2:])
        self.assertEqual(filter_entities(values, [], "all_mapped")[0], [])
        with self.assertRaises(ValueError):
            filter_entities(values, [], "unknown")


if __name__ == "__main__":
    unittest.main()
