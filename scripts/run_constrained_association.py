"""Run relation review with a finite, exact-source-reference output grammar."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, write_json
from patientphex.llm_association_references import parse_decision, prepare_prompt
from patientphex.llm_review_grammar import ReviewGrammar
from scripts.run_llm_association import atomic_json, canonical_hash, load_plan, load_tasks, prepare
from scripts.run_llm_reference_review import evaluate_pilot


def infer(plan_path, plan, fold, preflight=False):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    directory, manifest, tasks = load_tasks(plan_path, plan, fold)
    spec = json.loads((ROOT / plan["environment_spec"]).read_text())
    for name in ["environment_acceptance.json", "agent_acceptance.json"]:
        accepted = json.loads((ROOT / "reports/llm_extraction" / name).read_text())
        if accepted["status"] != "passed" or accepted["spec_sha256"] != canonical_hash(spec):
            raise ValueError("The unchanged runtime lacks matching acceptance evidence.")
    if torch.__version__ != spec["torch"] or transformers.__version__ != spec["transformers"]:
        raise ValueError("Inference was launched in a different runtime.")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / spec["model_path"], local_files_only=True, trust_remote_code=False)
    prompts = [prepare_prompt(tokenizer, task, spec["maximum_context_tokens"], plan["generation"]["max_new_tokens"]) for task in tasks]
    grammars = {count: ReviewGrammar(tokenizer, count) for count in {len(prompt["fragments"]) for prompt in prompts}}
    maximum_grammar_tokens = max(grammar.maximum_tokens for grammar in grammars.values())
    if maximum_grammar_tokens > plan["generation"]["max_new_tokens"]:
        raise ValueError("A valid grammar completion could be cut by the output budget.")
    audit = {**manifest, "max_input_tokens": max(prompt["input_tokens"] for prompt in prompts),
             "complete_evidence_count": sum(prompt["complete_evidence"] for prompt in prompts),
             "dropped_fragments": sum(prompt["dropped_fragments"] for prompt in prompts),
             "hard_nonthinking": True, "protocol": "finite_json_source_references", "maximum_grammar_tokens": maximum_grammar_tokens,
             "allowed_completion_counts": {str(count): len(grammar.responses) for count, grammar in grammars.items()}}
    write_json(directory / "preflight.json", audit)
    print(json.dumps({"event": "prompts_prepared", **audit}), flush=True)
    if preflight:
        return
    if not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "2"}:
        raise ValueError("An explicitly authorized CUDA device is required.")
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(2)
    cache = directory / "prompts"
    cache.mkdir(exist_ok=True)
    model, records = None, []
    for number, (task, prompt) in enumerate(zip(tasks, prompts, strict=True)):
        grammar = grammars[len(prompt["fragments"])]
        grammar_hash = canonical_hash(grammar.sequences)
        identity = canonical_hash({"messages": prompt["messages"], "generation": plan["generation"],
                                   "model_manifest_sha256": spec["model_manifest_sha256"], "grammar_sha256": grammar_hash})
        path = cache / f"{task['task_id']}.json"
        if path.exists():
            raw = json.loads(path.read_text())
            if raw["prompt_identity"] != identity or raw["task_id"] != task["task_id"]:
                raise ValueError("Cached prompt or grammar differs from this source input.")
            if hashlib.sha256(raw["response"].encode()).hexdigest() != raw["response_sha256"]:
                raise ValueError("Cached response is corrupted.")
        else:
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(ROOT / spec["model_path"], local_files_only=True, trust_remote_code=False,
                    torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation=spec["attention"]).eval()
            inputs = tokenizer(prompt["text"], add_special_tokens=False, return_tensors="pt").to("cuda")
            step = time.time()
            with torch.inference_mode():
                generated = model.generate(**inputs, **plan["generation"], temperature=None, top_p=None, top_k=None,
                    prefix_allowed_tokens_fn=grammar.callback(inputs["input_ids"].shape[1]),
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
            output_ids = generated[0, inputs["input_ids"].shape[1]:]
            sequence = output_ids.tolist()
            if sequence not in grammar.sequences:
                raise ValueError("Generated tokens did not match any complete legal response.")
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            raw = {"task_id": task["task_id"], "prompt_identity": identity, "response": response,
                   "response_sha256": hashlib.sha256(response.encode()).hexdigest(), "grammar_sha256": grammar_hash,
                   "input_tokens": prompt["input_tokens"], "output_tokens": len(output_ids),
                   "hit_token_limit": len(output_ids) == plan["generation"]["max_new_tokens"], "elapsed_seconds": time.time()-step}
            atomic_json(path, raw)
        if raw["response"] not in grammar.responses:
            raise ValueError("Saved response is outside the finite output grammar.")
        parsed = parse_decision(raw["response"], prompt["fragments"])
        records.append({**raw, **parsed, "parse_error": None, "pmc_id": task["pmc_id"],
                        "patient_id": task["patient_id"], "concept": task["concept"],
                        "complete_evidence": prompt["complete_evidence"], "dropped_fragments": prompt["dropped_fragments"],
                        "retained_fragments": prompt["fragments"]})
        progress = {"event": "task_complete", "fold": fold, "completed": number+1, "total": len(tasks),
                    "format_errors": 0, "decisions": dict(Counter(record["decision"] for record in records)),
                    "elapsed_seconds": time.time()-started}
        atomic_json(directory / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    write_json(directory / "decisions.json", records)
    summary = {**audit, "status": "completed", "decisions_sha256": digest_file(directory / "decisions.json"),
               "format_success_rate": 1.0, "format_errors": 0,
               "elapsed_seconds": time.time()-started, "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
               "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
               "generated_tokens": sum(record["output_tokens"] for record in records),
               "generation_seconds": sum(record["elapsed_seconds"] for record in records),
               "decisions": dict(Counter(record["decision"] for record in records))}
    write_json(directory / "summary.json", summary)
    print(json.dumps({"event": "completed", **summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "preflight", "infer", "evaluate"])
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/llm_association_constrained/plan.json")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.stage == "prepare":
        prepare(args.plan, plan)
    elif args.stage == "evaluate":
        evaluate_pilot(args.plan, plan)
    else:
        infer(args.plan, plan, args.fold, preflight=args.stage == "preflight")
