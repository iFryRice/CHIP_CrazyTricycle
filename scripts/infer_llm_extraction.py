"""Run fold-isolated Qwen extraction with resumable, source-bound prompt records."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.llm_extraction import DemonstrationRetriever, budget_prompt, parse_quotes, prompt_identity, source_chunks
from scripts.optimize_cpu_association import blind


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def run(plan_path, fold):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    plan = json.loads(plan_path.read_text())
    if fold not in plan["pilot_folds"]:
        raise ValueError("This plan authorizes pilot folds only.")
    for name, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"Frozen input/code changed: {name}")
    spec = json.loads((ROOT / "experiments/llm_extraction/environment.json").read_text())
    acceptance = json.loads((ROOT / "reports/llm_extraction/environment_acceptance.json").read_text())
    if acceptance["status"] != "passed" or acceptance["spec_sha256"] != canonical_hash(spec):
        raise ValueError("The declared environment lacks a matching acceptance witness.")
    agent_acceptance = json.loads((ROOT / "reports/llm_extraction/agent_acceptance.json").read_text())
    if agent_acceptance["status"] != "passed" or agent_acceptance["spec_sha256"] != canonical_hash(spec):
        raise ValueError("The fresh-agent runbook acceptance is missing.")
    if torch.__version__ != spec["torch"] or transformers.__version__ != spec["transformers"]:
        raise ValueError("Inference was launched in a different runtime.")
    if not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "2"}:
        raise ValueError("An explicitly authorized CUDA device is required.")
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(2)
    documents = read_jsonl(ROOT / plan["train_path"])
    splits = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    all_ids = {str(document["pmc_id"]) for document in documents}
    if sorted(identifier for values in splits for identifier in values) != sorted(all_ids):
        raise ValueError("The fixed folds do not partition the training corpus.")
    validation_ids = set(splits[fold])
    training = [document for document in documents if str(document["pmc_id"]) not in validation_ids]
    validation = [blind(document) for document in documents if str(document["pmc_id"]) in validation_ids]
    retriever = DemonstrationRetriever().fit(training, validation_ids)
    chunks = [chunk for document in validation for chunk in source_chunks(document)]
    if len(chunks) != plan["expected_chunks"][str(fold)]:
        raise ValueError("The declared source chunk coverage changed.")
    output = ROOT / plan["work_dir"] / f"fold{fold}"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"plan_sha256": digest_file(plan_path), "fold": fold,
                "training_document_ids": sorted(retriever.training_ids), "validation_document_ids": sorted(validation_ids),
                "chunk_ids": [chunk["chunk_id"] for chunk in chunks], "answers_removed_before_windowing": True}
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Saved inference belongs to different source data or code.")
    write_json(identity_path, identity)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / spec["model_path"], local_files_only=True, trust_remote_code=False)
    prepared = []
    for chunk in chunks:
        examples = retriever.retrieve(chunk, count=plan["demonstrations"])
        prompt = budget_prompt(tokenizer, chunk, examples, spec["maximum_context_tokens"], plan["generation"]["max_new_tokens"])
        if set(prompt["demonstration_document_ids"]) & validation_ids:
            raise ValueError("A held-out answer entered a demonstration.")
        prompt["identity"] = prompt_identity(prompt["messages"], plan["generation"], spec["model_manifest_sha256"])
        prepared.append((chunk, prompt))
    print(json.dumps({"event": "prompts_prepared", "fold": fold, "chunks": len(chunks),
                      "max_input_tokens": max(prompt["input_tokens"] for _, prompt in prepared)}), flush=True)
    cache_dir = output / "prompts"
    cache_dir.mkdir(exist_ok=True)
    model = None
    records, responses, format_errors = {}, [], 0
    for number, (chunk, prompt) in enumerate(prepared):
        cache_path = cache_dir / f"{number:05d}.json"
        if cache_path.exists():
            record = json.loads(cache_path.read_text())
            if record["prompt_identity"] != prompt["identity"] or record["chunk_id"] != chunk["chunk_id"]:
                raise ValueError("Cached prompt does not match this exact input.")
            if record["response_sha256"] != hashlib.sha256(record["response"].encode()).hexdigest():
                raise ValueError("Cached model response is corrupted.")
        else:
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(ROOT / spec["model_path"], local_files_only=True, trust_remote_code=False,
                    torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation=spec["attention"]).eval()
            inputs = tokenizer(prompt["text"], return_tensors="pt", add_special_tokens=False).to("cuda")
            step_started = time.time()
            with torch.inference_mode():
                generated = model.generate(**inputs, **plan["generation"], temperature=None, top_p=None, top_k=None,
                                           pad_token_id=tokenizer.eos_token_id)
            output_ids = generated[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            record = {"prompt_identity": prompt["identity"], "chunk_id": chunk["chunk_id"], "pmc_id": chunk["pmc_id"],
                      "source_sha256": hashlib.sha256(chunk["text"].encode()).hexdigest(),
                      "demonstration_chunk_ids": prompt["demonstration_chunk_ids"],
                      "demonstration_document_ids": prompt["demonstration_document_ids"],
                      "dropped_demonstrations": prompt["dropped_demonstrations"], "input_tokens": prompt["input_tokens"],
                      "output_tokens": len(output_ids), "hit_token_limit": len(output_ids) == plan["generation"]["max_new_tokens"],
                      "response": response, "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                      "elapsed_seconds": time.time() - step_started}
            temporary = cache_path.with_suffix(".tmp")
            write_json(temporary, record)
            temporary.replace(cache_path)
        try:
            spans, rejected = parse_quotes(record["response"], chunk)
            parse_error = None
        except (ValueError, TypeError) as error:
            spans, rejected, parse_error = [], [], type(error).__name__
            format_errors += 1
        for span in spans:
            records.setdefault(chunk["pmc_id"], {})[(span["offset"], span["length"])] = {**span, "scores": {"positive": 1.0, "NO": 0.0}}
        responses.append({**record, "accepted_quotes": len(spans), "rejected_quotes": rejected, "parse_error": parse_error})
        print(json.dumps({"event": "chunk_complete", "fold": fold, "completed": number + 1, "total": len(chunks),
                          "accepted": len(spans), "format_errors": format_errors, "elapsed_seconds": time.time() - started}), flush=True)
    predictions = [{"pmc_id": document["pmc_id"], "spans": [value for _, value in sorted(records.get(str(document["pmc_id"]), {}).items())]} for document in validation]
    write_jsonl(output / "spans.jsonl", predictions)
    write_json(output / "prompt_audit.json", responses)
    result = {**identity, "status": "completed", "span_sha256": digest_file(output / "spans.jsonl"),
              "prompt_audit_sha256": digest_file(output / "prompt_audit.json"), "format_errors": format_errors,
              "format_success_rate": 1 - format_errors / len(chunks), "elapsed_seconds": time.time() - started,
              "raw_mentions": sum(len(document["spans"]) for document in predictions),
              "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "generated_tokens": sum(item["output_tokens"] for item in responses),
              "generation_seconds": sum(item["elapsed_seconds"] for item in responses),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
    write_json(output / "summary.json", result)
    print(json.dumps({"event": "completed", **result}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/llm_extraction/plan.json")
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    run(args.plan, args.fold)
