"""Linking and streaming-crosswalk contracts without external data or models."""

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from patientphex.ontology import Ontology
from patientphex.span_linking import SpanLinker
from scripts.export_umls_hpo_terms import allowed_hpo_ids, export_terms
from tests.test_entities import OBO, document


def annotated(text, identifier, pmc_id="train"):
    return document(text, entities=[{"text": text, "offset": 0, "length": len(text), "identifier": identifier, "note": None}], pmc_id=pmc_id)


def prediction(text, offset=0, scores=None, pmc_id="test"):
    scores = scores or {"positive": 0.9}
    return {"pmc_id": pmc_id, "spans": [{"offset": offset, "length": len(text), "text": text,
                                        "labels": list(scores), "scores": scores}]}


class SpanLinkingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.obo = self.root / "hp.obo"
        self.obo.write_text(OBO, encoding="utf-8")
        self.ontology = Ontology(self.obo)

    def tearDown(self):
        self.temporary.cleanup()

    def make_export(self):
        source = self.root / "terms.sqlite"
        db = sqlite3.connect(source)
        db.execute("CREATE TABLE sources(vsab TEXT,source_version TEXT,rsab TEXT)")
        db.execute("INSERT INTO sources VALUES('HPO2025_11_24','2025_11_24','HPO')")
        db.execute("CREATE TABLE terms(term TEXT,cui TEXT,sab TEXT,code TEXT,aui TEXT,tty TEXT,is_pref TEXT,lat TEXT,suppress TEXT)")
        db.executemany("INSERT INTO terms VALUES(?,?,?,?,?,?,?,?,?)", [
            ("Seizure", "C1", "HPO", "HP:0001250", "A1", "PT", "Y", "ENG", "N"),
            ("convulsive episode", "C1", "SNOMEDCT_US", "S1", "A2", "PT", "Y", "ENG", "N"),
            ("ambiguous feature", "C1", "MSH", "M1", "A3", "SY", "N", "ENG", "N"),
            ("Short stature", "C2", "HPO", "HP:0004322", "A4", "PT", "Y", "ENG", "N"),
            ("ambiguous feature", "C2", "MSH", "M2", "A5", "SY", "N", "ENG", "N"),
            ("unrelated", "C3", "HPO", "HP:0000002", "A6", "PT", "Y", "ENG", "N"),
            ("disallowed source", "C1", "UNKNOWN", "X", "A7", "PT", "Y", "ENG", "N"),
            ("nonenglish", "C1", "MSH", "M3", "A8", "PT", "Y", "FRE", "N"),
            ("suppressed", "C1", "MSH", "M4", "A9", "PT", "Y", "ENG", "Y"),
        ])
        db.commit()
        db.close()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        path = self.root / "crosswalk.tsv.gz"
        stats = export_terms(source, self.obo, path, batch_size=1)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
        return path, stats

    def test_export_is_filtered_streamed_and_read_only(self):
        version, allowed = allowed_hpo_ids(self.obo)
        self.assertEqual(version, self.ontology.version)
        self.assertEqual(allowed, self.ontology.allowed_ids)
        path, stats = self.make_export()
        self.assertEqual(stats["exported_rows"], 5)
        self.assertEqual(stats["anchor_cuis"], 2)
        self.assertEqual(stats["covered_hpo_ids"], 2)
        linker = SpanLinker(self.ontology, path)
        self.assertEqual(linker.resolve("convulsive episode")["identifier"], "HP:0001250")
        self.assertEqual(linker.resolve("suppressed")["identifier"], "-1")
        self.assertEqual(linker.resolve("nonenglish")["identifier"], "-1")
        self.assertEqual(linker.resolve("disallowed source")["identifier"], "-1")
        self.assertEqual(linker.resolve("unrelated")["identifier"], "-1")

    def test_umls_ambiguity_remains_unmapped_not_compound(self):
        path, _ = self.make_export()
        result = SpanLinker(self.ontology, path).resolve("ambiguous feature")
        self.assertEqual(result["identifier"], "-1")
        self.assertTrue(result["ambiguous"])
        self.assertEqual(result["candidates"], ["HP:0001250", "HP:0004322"])

    def test_fit_reset_forbidden_ids_and_annotation_blind_prediction(self):
        linker = SpanLinker(self.ontology).fit([annotated("tiny build", "HP:0004322")], forbidden_document_ids=["validation"])
        target = document("tiny build", pmc_id="validation", entities=[{"identifier": "HP:0001250", "text": "tiny build"}])
        entities, _ = linker.link_document(target, prediction("tiny build", pmc_id="validation"))
        self.assertEqual(entities[0]["identifier"], "HP:0004322")
        linker.fit([])
        self.assertEqual(linker.resolve("tiny build")["identifier"], "-1")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            linker.fit([annotated("tiny build", "HP:0004322", pmc_id="validation")], forbidden_document_ids=["validation"])

    def test_document_majority_not_repeated_mentions_breaks_training_ties(self):
        first = annotated("alias", "HP:0001250", pmc_id="first")
        first["entities"] *= 100
        second = annotated("alias", "HP:0004322", pmc_id="second")
        linker = SpanLinker(self.ontology).fit([first, second])
        self.assertTrue(linker.resolve("alias")["ambiguous"])
        linker.fit([first, second, annotated("alias", "HP:0001250", pmc_id="third")])
        self.assertEqual(linker.resolve("alias")["identifier"], "HP:0001250")

    def test_compound_unmapped_and_invalid_training_ids(self):
        linker = SpanLinker(self.ontology).fit([
            annotated("compound feature", "HP:0001250;HP:9999998"),
            annotated("novel phenotype", "-1", pmc_id="other"),
            annotated("bad feature", "HP:0000118", pmc_id="bad"),
        ])
        self.assertEqual(linker.resolve("compound feature")["identifier"], "HP:0001250;HP:0004322")
        self.assertEqual(linker.resolve("novel phenotype")["identifier"], "-1")
        self.assertEqual(len(linker.fit_audit["invalid_identifier_annotations"]), 1)
        self.assertEqual(linker.resolve("bad feature")["source"], "no_candidate")

    def test_unicode_offsets_negation_conflicts_and_input_immutability(self):
        source = document("🧬 Seizures", offset=100)
        record = prediction("Seizures", offset=102, scores={"positive": 0.9, "NO": 0.95})
        original = copy.deepcopy((source, record))
        entities, audit = SpanLinker(self.ontology).link_document(source, record, thresholds={"positive": 0.9, "NO": 0.95})
        self.assertEqual(entities, [{"identifier": "HP:0001250", "type": "Phenotype", "offset": 102,
                                     "length": 8, "text": "Seizures", "note": "NO"}])
        self.assertEqual(len(audit["label_conflicts"]), 1)
        self.assertEqual((source, record), original)
        record["spans"][0]["scores"] = {"positive": 0.95, "NO": 0.95}
        self.assertIsNone(SpanLinker(self.ontology).link_document(source, record)[0][0]["note"])

    def test_bad_offsets_scores_duplicates_and_cache_tampering_fail(self):
        linker = SpanLinker(self.ontology)
        with self.assertRaisesRegex(ValueError, "thresholds"):
            linker.link_document(document("Seizure"), prediction("Seizure"), thresholds={})
        with self.assertRaisesRegex(ValueError, "original text"):
            linker.link_document(document("Seizure"), prediction("Seizure", offset=1))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            linker.link_document(document("Seizure"), prediction("Seizure", scores={"positive": float("nan")}))
        repeated = prediction("Seizure")
        repeated["spans"] *= 2
        with self.assertRaisesRegex(ValueError, "repeated spans"):
            linker.link_document(document("Seizure"), repeated)
        path, _ = self.make_export()
        manifest_path = path.with_name(path.name + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["output_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "hash"):
            SpanLinker(self.ontology, path)

    def test_knowledge_spans_use_umls_but_never_training_aliases(self):
        path, _ = self.make_export()
        linker = SpanLinker(self.ontology, path)
        source = document("No convulsive episode, but Seizures and secret alias.", offset=200)
        before = linker.knowledge_spans(source)
        linker.fit([annotated("secret alias", "HP:0004322")])
        after = linker.knowledge_spans(source)
        self.assertEqual(before, after)
        self.assertEqual([s["text"] for s in after["spans"]], ["convulsive episode", "Seizures"])
        self.assertEqual(after["spans"][0]["offset"], 203)
        self.assertEqual(after["spans"][0]["scores"], {"NO": 1.0})
        self.assertEqual(after["spans"][1]["scores"], {"positive": 1.0})

    def test_knowledge_keeps_nested_spans_and_uppercase_short_forms(self):
        self.obo.write_text(OBO + '\n[Term]\nid: HP:1234567\nname: Growth problem\nsynonym: "SS" EXACT []\n'
                            'synonym: "stature" EXACT []\nis_a: HP:0000118\n', encoding="utf-8")
        linker = SpanLinker(Ontology(self.obo))
        record = linker.knowledge_spans(document("Short stature SS ss"))
        self.assertEqual([span["text"] for span in record["spans"]], ["Short stature", "stature", "SS"])


if __name__ == "__main__":
    unittest.main()
