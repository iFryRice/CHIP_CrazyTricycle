"""Pinned SapBERT retrieval over fixed HPO/UMLS names and conservative merging."""

from collections import defaultdict
import copy
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .data import digest_file, write_json


def fixed_names(ontology, umls_path):
    names = set()
    for identifier in sorted(ontology.allowed_ids):
        term = ontology.terms[identifier]
        for name in [term["name"], *term["exact_synonyms"]]:
            if name.strip():
                names.add((identifier, name.strip()))
    with gzip.open(umls_path, "rt", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["hpo_id"] in ontology.allowed_ids and row["term"].strip():
                names.add((row["hpo_id"], row["term"].strip()))
    return sorted(names)


class SemanticIndex:
    """Use CLS vectors as specified by the original SapBERT model card."""

    def __init__(self, model_path, cache_dir, ontology, umls_path, identity, batch_size=128):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("The declared GPU inference route is unavailable.")
        torch.manual_seed(20260929)
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, attn_implementation="eager").to(self.device).eval()
        names = fixed_names(ontology, umls_path)
        self.concepts = sorted({identifier for identifier, _ in names})
        concept_index = {identifier: index for index, identifier in enumerate(self.concepts)}
        self.term_concepts = torch.tensor([concept_index[identifier] for identifier, _ in names], dtype=torch.long, device=self.device)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"identity": identity, "term_count": len(names), "concept_count": len(self.concepts), "max_tokens": 64, "pooling": "CLS_L2"}
        meta_path, vectors_path = cache_dir / "index.json", cache_dir / "vectors.npy"
        if meta_path.exists() or vectors_path.exists():
            if not meta_path.exists() or not vectors_path.exists():
                raise ValueError("Incomplete semantic index; inspect it before rebuilding.")
            saved = json.loads(meta_path.read_text())
            if saved["configuration"] != metadata or saved["vectors_sha256"] != digest_file(vectors_path):
                raise ValueError("Semantic index identity or content hash changed.")
            vectors = np.load(vectors_path, allow_pickle=False)
        else:
            vectors = self.encode([name for _, name in names])
            np.save(vectors_path, vectors, allow_pickle=False)
            write_json(meta_path, {"configuration": metadata, "vectors_sha256": digest_file(vectors_path)})
        if vectors.shape != (len(names), 768) or not np.isfinite(vectors).all():
            raise ValueError("Malformed semantic vectors.")
        self.vectors = torch.as_tensor(vectors, dtype=torch.float32, device=self.device)
        self.audit = {**metadata, "vectors_sha256": digest_file(vectors_path), "device_name": torch.cuda.get_device_name(0)}

    def encode(self, strings):
        torch = self.torch
        if not strings:
            return np.empty((0, 768), dtype=np.float32)
        output = []
        with torch.inference_mode():
            for start in range(0, len(strings), self.batch_size):
                batch = self.tokenizer(strings[start:start + self.batch_size], padding=True, truncation=True, max_length=64, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hidden = self.model(**batch).last_hidden_state[:, 0, :]
                normalized = torch.nn.functional.normalize(hidden.float(), dim=1)
                if not torch.isfinite(normalized).all():
                    raise ValueError("Nonfinite semantic embedding.")
                output.append(normalized.cpu().numpy())
                if start % (self.batch_size * 50) == 0:
                    print(f"encoded={min(start+self.batch_size,len(strings))}/{len(strings)}", flush=True)
        return np.concatenate(output)

    def retrieve(self, strings):
        torch = self.torch
        vectors = self.encode(strings)
        result = {}
        with torch.inference_mode():
            for start in range(0, len(strings), self.batch_size):
                batch = torch.as_tensor(vectors[start:start + self.batch_size], device=self.device)
                similarities = batch @ self.vectors.T
                # Aggregate by concept before top-k; many synonyms of one concept
                # must not hide the best competing concept or inflate its margin.
                concept_scores = torch.full((len(batch), len(self.concepts)), -2.0, device=self.device)
                concept_scores.scatter_reduce_(1, self.term_concepts.expand(len(batch), -1), similarities, reduce="amax", include_self=True)
                values, indices = concept_scores.topk(3, dim=1)
                for text, scores, ids in zip(strings[start:start + self.batch_size], values.cpu().tolist(), indices.cpu().tolist(), strict=True):
                    result[text] = {"identifier": self.concepts[ids[0]], "similarity": scores[0], "margin": scores[0] - scores[1], "alternatives": [{"identifier": self.concepts[i], "similarity": score} for i, score in zip(ids, scores, strict=True)]}
        return result


def merge_neural_additions(base_entities, candidates, span_threshold, semantic_threshold, margin_threshold=0.02):
    """Preserve the baseline and arbitrate new overlaps by evidence strength."""
    selected = []
    for item in candidates:
        if item["span_score"] < span_threshold:
            continue
        if item["source"] == "semantic" and (item["similarity"] < semantic_threshold or item["margin"] < margin_threshold):
            continue
        if item["entity"]["identifier"] == "-1":
            continue
        selected.append(item)
    accepted = copy.deepcopy(base_entities)
    for item in sorted(selected, key=lambda item: (-item["span_score"] * item["similarity"], item["entity"]["length"], item["entity"]["offset"], item["entity"]["identifier"])):
        entity = item["entity"]
        if any(entity["offset"] < other["offset"] + other["length"] and other["offset"] < entity["offset"] + entity["length"] for other in accepted):
            continue
        accepted.append(copy.deepcopy(entity))
    return sorted(accepted, key=lambda entity: (entity["offset"], entity["length"]))
