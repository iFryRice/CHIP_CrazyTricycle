"""Contract tests for offset preservation and training-only lexical learning."""

from pathlib import Path
import tempfile
import unittest

from patientphex.entities import EntityExtractor, is_negated
from patientphex.ontology import Ontology


OBO = '''format-version: 1.2
data-version: hp/releases/2026-06-23

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
synonym: "Epileptic seizure" EXACT []
is_a: HP:0000118

[Term]
id: HP:0004322
name: Short stature
alt_id: HP:9999998
is_a: HP:0000118

[Term]
id: HP:0000002
name: Unrelated term

[Term]
id: HP:9999999
name: Old term
is_a: HP:0000118
is_obsolete: true
'''


def document(text, offset=0, entities=None, pmc_id="test"):
    return {"pmc_id": pmc_id, "full_text": [{"text": text, "offset": offset,
            "section_type": "CASE", "type": "paragraph"}], "entities": entities or []}


class EntityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "hp.obo"
        self.path.write_text(OBO, encoding="utf-8")
        self.ontology = Ontology(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_ontology_branch_and_compound_identifiers(self):
        self.assertEqual(self.ontology.version, "2026-06-23")
        self.assertEqual(self.ontology.allowed_ids, {"HP:0001250", "HP:0004322"})
        self.assertEqual(self.ontology.normalize_identifier("HP:9999998;-1"), "HP:0004322;-1")
        self.assertIsNone(self.ontology.normalize_identifier("HP:0000002"))

    def test_non_bmp_and_hyphenated_plural_offsets(self):
        text = "🧬 Patient had epileptic-seizures and short stature."
        extractor = EntityExtractor(self.ontology, mode="dictionary")
        entities = extractor.predict(document(text, offset=100))
        self.assertEqual([e["text"] for e in entities], ["epileptic-seizures", "short stature"])
        for entity in entities:
            start = entity["offset"] - 100
            self.assertEqual(text[start:start + entity["length"]], entity["text"])
        self.assertEqual(entities[0]["offset"], 114)

    def test_word_boundaries_and_sentence_boundaries(self):
        extractor = EntityExtractor(self.ontology, mode="dictionary")
        self.assertEqual(extractor.predict(document("pseudoseizure. short. stature")), [])

    def test_negation_and_contrast(self):
        text = "No seizures, but short stature was present."
        entities = EntityExtractor(self.ontology, mode="dictionary").predict(document(text))
        self.assertEqual([e["note"] for e in entities], ["NO", None])
        self.assertFalse(is_negated("Not only seizures", 9, 17))
        self.assertTrue(is_negated("Seizures were absent.", 0, 8))

    def test_training_alias_preserves_compound_and_unmapped_ids(self):
        text = "unusual gait and tiny build"
        gold = [
            {"text": "unusual gait", "offset": 0, "length": 12, "identifier": "-1", "note": None},
            {"text": "tiny build", "offset": 17, "length": 10,
             "identifier": "HP:0004322;HP:0001250", "note": None},
        ]
        extractor = EntityExtractor(self.ontology, mode="dictionary").fit([document(text, entities=gold)])
        self.assertEqual([e["identifier"] for e in extractor.predict(document(text))],
                         ["-1", "HP:0004322;HP:0001250"])
        extractor.fit([])
        self.assertEqual(extractor.predict(document(text)), [])

    def test_document_local_abbreviation(self):
        text = "Short stature (SS) was observed. SS persisted."
        entities = EntityExtractor(self.ontology, mode="dictionary").predict(document(text))
        self.assertEqual([e["text"] for e in entities], ["Short stature", "SS", "SS"])
        self.assertTrue(all(e["identifier"] == "HP:0004322" for e in entities))

    def test_training_abbreviation_does_not_match_lowercase_word(self):
        gold = [{"text": "AS", "offset": 0, "length": 2,
                 "identifier": "HP:0004322", "note": None}]
        extractor = EntityExtractor(self.ontology, mode="dictionary").fit(
            [document("AS", entities=gold)]
        )
        entities = extractor.predict(document("as AS"))
        self.assertEqual([entity["text"] for entity in entities], ["AS"])

    def test_training_frequency_disambiguates_dictionary_alias(self):
        self.path.write_text(OBO + '\n[Term]\nid: HP:1234567\nname: Seizure\n'
                            'is_a: HP:0000118\n', encoding="utf-8")
        ontology = Ontology(self.path)
        gold = [{"text": "Seizure", "offset": 0, "length": 7,
                 "identifier": "HP:1234567", "note": None}]
        extractor = EntityExtractor(ontology, mode="dictionary").fit(
            [document("Seizure", entities=gold)]
        )
        self.assertEqual(extractor.predict(document("Seizures"))[0]["identifier"], "HP:1234567")


if __name__ == "__main__":
    unittest.main()
