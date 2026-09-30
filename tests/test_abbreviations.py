"""Document definitions must never become global aliases or change offsets."""

import copy
import unittest

from patientphex.abbreviations import apply_definitions, extract_definitions, resolve_definitions


def document(text, offset=0):
    return {"full_text": [{"offset": offset, "text": text}], "entities": [], "association": []}


class Lookup:
    def __init__(self, mapping):
        self.mapping = mapping

    def resolve(self, text):
        return {"identifier": self.mapping.get(text, "-1"), "ambiguous": False,
                "source": "hpo_exact" if text in self.mapping else "no_candidate"}


class AbbreviationTests(unittest.TestCase):
    def test_forward_and_reverse_alignment_preserve_global_offsets(self):
        text = "She had sensorineural hearing loss (SNHL). MODY (Maturity-Onset Diabetes of the Young) was considered."
        definitions = extract_definitions(document(text, 100))
        self.assertEqual([(d["short"], d["long"]) for d in definitions],
                         [("SNHL", "sensorineural hearing loss"), ("MODY", "Maturity-Onset Diabetes of the Young")])
        for definition in definitions:
            start = definition["long_offset"] - 100
            self.assertEqual(text[start:start + definition["long_length"]], definition["long"])
            start = definition["short_offset"] - 100
            self.assertEqual(text[start:start + len(definition["short"])], definition["short"])

    def test_conflicting_definitions_abstain_and_do_not_leak_between_documents(self):
        linker = Lookup({"central nervous system": "HP:1", "congenital nephrotic syndrome": "HP:2"})
        resolved = resolve_definitions(document("central nervous system (CNS). congenital nephrotic syndrome (CNS)."), linker)
        self.assertEqual(resolved["CNS"]["status"], "abstain")
        self.assertEqual(resolve_definitions(document("CNS is discussed."), linker), {})

    def test_correction_preserves_original_note_and_additions_match_original_text(self):
        source = document("sensorineural hearing loss (SNHL). No SNHL. SNHL was noted.", 50)
        resolved = resolve_definitions(source, Lookup({"sensorineural hearing loss": "HP:1"}))
        start = source["full_text"][0]["text"].index("SNHL") + 50
        entity = {"offset": start, "length": 4, "text": "SNHL", "identifier": "HP:2", "note": "NO", "type": "Phenotype"}
        original = copy.deepcopy(entity)
        output, changes = apply_definitions(source, [entity], resolved, add_missing=True)
        self.assertEqual(output[0]["identifier"], "HP:1")
        self.assertEqual(output[0]["note"], "NO")
        self.assertEqual(entity, original)
        self.assertEqual(len(output), 3)
        self.assertEqual(output[1]["note"], "NO")
        self.assertEqual(len(changes), 3)

    def test_unknown_definition_suppression_is_explicit_and_no_gold_is_read(self):
        source = document("central nervous system (CNS)")
        resolved = resolve_definitions(source, Lookup({}))
        entity = {"offset": 24, "length": 3, "text": "CNS", "identifier": "HP:1", "note": None}
        self.assertEqual(apply_definitions(source, [entity], resolved)[0], [entity])
        self.assertEqual(apply_definitions(source, [entity], resolved, suppress_unresolved=True)[0], [])
        with self.assertRaisesRegex(ValueError, "blind"):
            apply_definitions({**source, "entities": [entity]}, [entity], resolved)

    def test_parenthetical_measurements_and_ordinary_words_are_not_definitions(self):
        self.assertEqual(extract_definitions(document("Height (120 cm), age (10 years), treatment (oral) and Fig. (2A).")), [])


if __name__ == "__main__":
    unittest.main()
