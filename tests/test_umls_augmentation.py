"""UMLS ambiguity and frozen span preservation."""

import copy
import gzip
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from patientphex.umls_augmentation import augment_entities, load_unambiguous_aliases


class UmlsAugmentationTests(unittest.TestCase):
    def test_excludes_ambiguous_short_and_out_of_branch_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terms.tsv.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write("term\thpo_id\nseizures\tHP:1\nseizures\tHP:2\nshort stature\tHP:1\nshort stature\tHP:1\nother phrase\tHP:3\nABC\tHP:1\n")
            aliases = load_unambiguous_aliases(path, SimpleNamespace(allowed_ids={"HP:1", "HP:2"}))
        self.assertEqual(aliases, {("short", "stature"): "HP:1"})

    def test_preserves_original_span_when_new_match_overlaps(self):
        base = [{"offset": 10, "length": 5, "identifier": "HP:1", "note": "NO"}]
        additions = [{"score": 0.99, "entity": {"offset": 8, "length": 12, "identifier": "HP:2"}}, {"score": 0.6, "entity": {"offset": 20, "length": 4, "identifier": "HP:3"}}]
        original = copy.deepcopy((base, additions))
        result = augment_entities(base, additions, 0.6)
        self.assertEqual(result, [base[0], additions[1]["entity"]])
        self.assertEqual((base, additions), original)

    def test_rejects_bad_threshold_and_deduplicates_additions(self):
        entity = {"offset": 1, "length": 5, "identifier": "HP:1"}
        self.assertEqual(augment_entities([], [{"score": 0.5, "entity": entity}] * 2, 0.5), [entity])
        with self.assertRaises(ValueError):
            augment_entities([], [], -0.1)


if __name__ == "__main__":
    unittest.main()
