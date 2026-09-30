"""Exercise source fidelity and protected review decisions for nested additions."""

import unittest
from patientphex.overlap_candidates import add_nested


def entity(start, text, note=None):
    return {"offset": start, "length": len(text), "text": text, "identifier": "HP:0000001", "note": note, "type": "Phenotype"}


def candidate(value):
    return {"entity": value, "span_score": 0.95, "similarity": 0.99, "margin": 0.1, "source": "semantic"}


class NestedCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.document = {"full_text": [{"offset": 10, "text": "short stature and delay"}]}
        self.old = entity(10, "short")
        self.new = entity(10, "short stature")

    def test_adds_containing_span_without_mutating_original(self):
        values, additions = add_nested(self.document, [self.old], [self.old], [candidate(self.new)])
        self.assertEqual(additions, [self.new])
        self.assertEqual(values, [self.old, self.new])
        values[0]["text"] = "changed"
        self.assertEqual(self.old["text"], "short")

    def test_never_reintroduces_reviewed_span_or_crosses_negation(self):
        _, additions = add_nested(self.document, [self.old, self.new], [self.old], [candidate(self.new)])
        self.assertEqual(additions, [])
        negated = {**self.old, "note": "NO"}
        _, additions = add_nested(self.document, [negated], [negated], [candidate(self.new)])
        self.assertEqual(additions, [])
        crossing = entity(12, "ort stature")
        _, additions = add_nested(self.document, [self.old], [self.old], [candidate(crossing)])
        self.assertEqual(additions, [])

    def test_rejects_altered_source_and_labelled_input(self):
        with self.assertRaises(ValueError):
            add_nested(self.document, [self.old], [self.old], [candidate({**self.new, "text": "SHORT STATURE"})])
        with self.assertRaises(ValueError):
            add_nested({**self.document, "entities": [self.old]}, [self.old], [self.old], [])


if __name__ == "__main__":
    unittest.main()
