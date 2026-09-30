"""Combine frozen concept decisions and supervised association models."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.joint_review import changed_examples
from patientphex.llm_concept_review import eligible, parse_decision
from patientphex.ontology import Ontology
from patientphex.patient_linking import _check_provenance
from patientphex.relation_text import Collator, build_model, configure_tokenizer, encode, examples
from patientphex.span_linking import SpanLinker
from scripts.optimize_cpu_association import blind, bootstrap_delta, error_counts, from_scores, validate_file
from scripts.run_llm_association import load_plan, load_tasks
from scripts.run_relation_seed_ensemble import verified_scores


def filtered_entities(plan, fold, baseline, val_ids, ontology):
    path = ROOT / plan["concept_plans"][str(fold)]
    source_plan = load_plan(path)
    directory, manifest, tasks = load_tasks(path, source_plan, fold)
    summary = json.loads((directory/"summary.json").read_text())
    if (summary["status"] != "completed" or summary["plan_sha256"] != digest_file(path)
            or summary["decisions_sha256"] != digest_file(directory/"decisions.json")
            or manifest["validation_document_ids"] != val_ids):
        raise ValueError("Concept decisions are incomplete, changed or from another fold.")
    records = json.loads((directory/"decisions.json").read_text())
    indexed = {row["task_id"]:row for row in records}
    fixed = SpanLinker(ontology)
    expected = {(identifier,i) for identifier in val_ids for i,e in enumerate(baseline[identifier]["entities"]) if eligible(e,ontology,fixed)}
    if (len(indexed) != len(records) or set(indexed) != {t["task_id"] for t in tasks}
            or {(t["pmc_id"],t["entity_index"]) for t in tasks} != expected):
        raise ValueError("Concept-review coverage differs from the protected frozen scope.")
    rejected = {identifier:set() for identifier in val_ids}
    for task in tasks:
        record = indexed[task["task_id"]]
        if baseline[task["pmc_id"]]["entities"][task["entity_index"]] != task["entity"]:
            raise ValueError("A concept decision no longer refers to the baseline entity.")
        if (any(record[k] != task[k] for k in source_plan["identity_fields"])
                or record["retained_fragments"] != task["fragments"] or record["parse_error"] is not None
                or hashlib.sha256(record["response"].encode()).hexdigest() != record["response_sha256"]):
            raise ValueError("Concept evidence or response identity changed.")
        parsed = parse_decision(record["response"],task["fragments"])
        if parsed != {k:record[k] for k in ["decision","evidence_ids"]}:
            raise ValueError("Concept parser no longer replays the original decision.")
        if parsed["decision"] == "other_or_general": rejected[task["pmc_id"]].add(task["entity_index"])
    filtered = {identifier:[e for i,e in enumerate(baseline[identifier]["entities"]) if i not in rejected[identifier]] for identifier in val_ids}
    return filtered, {"concept_plan_sha256":digest_file(path),"decisions_sha256":summary["decisions_sha256"],
                      "format_success_rate":summary["format_success_rate"],"review_tasks":len(tasks),
                      "removed_entities":sum(len(v) for v in rejected.values())}


def inputs(plan):
    docs = {str(d["pmc_id"]):d for d in read_jsonl(ROOT/plan["train_path"])}
    baseline = {str(d["pmc_id"]):d for d in read_jsonl(ROOT/plan["baseline_oof"])}
    folds = json.loads((ROOT/plan["split_path"]).read_text())["folds"]
    if set(docs) != set(baseline) or sorted(sum(folds,[])) != sorted(docs): raise ValueError("Source/fold coverage changed.")
    return docs, baseline, folds


def infer(plan_path,plan,fold):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import transformers
    if fold not in plan["pilot_folds"]: raise ValueError("Undeclared joint-inference fold.")
    output = ROOT/plan["work_dir"]/f"fold{fold}"
    if output.exists(): raise FileExistsError("Preserve earlier joint inference.")
    if not torch.cuda.is_available() or torch.__version__ != "2.6.0+cu124" or transformers.__version__ != "4.49.0":
        raise RuntimeError("The accepted BiomedBERT inference environment changed.")
    torch.set_num_threads(2)
    started = time.time()
    docs,baseline,folds = inputs(plan)
    val_ids = folds[fold]
    ensemble = load_plan(ROOT/plan["ensemble_plan"])
    ontology = Ontology(ROOT/plan["ontology_path"])
    entities, concept_audit = filtered_entities(plan,fold,baseline,val_ids,ontology)
    with (ROOT/ensemble["association_models"][str(fold)]).open("rb") as handle: structural = pickle.load(handle)
    _check_provenance(structural.provenance_,set(docs)-set(val_ids),set(val_ids))
    context_scores = {}
    for identifier in val_ids:
        document,old = blind(docs[identifier]),baseline[identifier]
        if from_scores(document,old["entities"],structural.predict_scores(document,old["entities"]),.5) != old:
            raise ValueError("Unfiltered context baseline no longer replays exactly.")
        context_scores[identifier] = structural.predict_scores(document,entities[identifier])
    model_path = ROOT/plan["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    configure_tokenizer(tokenizer)
    new_rows = [encode(tokenizer,row) for identifier in val_ids for row in examples(blind(docs[identifier]),entities[identifier],ontology)]
    new_index = {r["task_id"]:r for r in new_rows}
    if len(new_index) != len(new_rows): raise ValueError("Duplicate filtered relation examples.")
    source_raw = None
    members, audits = [], []
    output.mkdir(parents=True)
    write_json(output/"filtered_entities.json",entities)
    write_json(output/"context_scores.json",context_scores)
    write_json(output/"examples.json",new_rows)
    for seed in ensemble["seeds"]:
        member = dict(ensemble["members"][str(seed)])
        if "fold_work_dirs" in member: member["work_dir"] = member["fold_work_dirs"][str(fold)]
        old_scores,evidence = verified_scores(member,fold,docs,val_ids,ensemble["parameters"])
        work = ROOT/member["work_dir"]/f"fold{fold}"
        manifest = json.loads((work/"manifest.json").read_text())
        raw_path = work/"validation_examples.json"
        if digest_file(raw_path) != manifest["validation_examples_sha256"]: raise ValueError("Original text examples changed.")
        raw = json.loads(raw_path.read_text())
        if source_raw is not None and raw != source_raw: raise ValueError("Seed members used different text representations.")
        source_raw = raw
        old_rows = [encode(tokenizer,row) for row in raw]
        changed = changed_examples(old_rows,new_rows)
        changed_ids = {r["task_id"] for r in changed}
        probabilities = {r["task_id"]:old_scores[r["pmc_id"],r["patient_id"],r["concept"]] for r in new_rows if r["task_id"] not in changed_ids}
        if changed:
            config = AutoConfig.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
            encoder = AutoModel.from_config(config,trust_remote_code=False,attn_implementation="eager")
            encoder.resize_token_embeddings(len(tokenizer),mean_resizing=False)
            model = build_model(encoder)
            checkpoint = torch.load(work/"last.pt",map_location="cpu",weights_only=True)
            if checkpoint["plan_sha256"] != member["fold_plan_sha256"][str(fold)] or checkpoint["smoke"] or checkpoint["fold"] != fold:
                raise ValueError("A joint-inference checkpoint differs from its declared member.")
            model.load_state_dict(checkpoint["model_state_dict"],strict=True); del checkpoint
            model.cuda().eval()
            with torch.inference_mode():
                for batch in DataLoader(changed,batch_size=12,shuffle=False,collate_fn=Collator(tokenizer)):
                    with torch.autocast(device_type="cuda",dtype=torch.float16):
                        values = model(**{k:v.cuda() for k,v in batch["inputs"].items()}).float().sigmoid()
                    if not torch.isfinite(values).all(): raise FloatingPointError("Nonfinite filtered-context probability.")
                    for row,value in zip(batch["rows"],values.cpu().tolist(),strict=True): probabilities[row["task_id"]]=value
            del model,encoder
            torch.cuda.empty_cache()
        if set(probabilities) != set(new_index): raise ValueError("Incomplete incremental inference.")
        write_json(output/f"seed{seed}_probabilities.json",probabilities)
        audits.append({"seed":seed,**evidence,"unchanged_input_reused":len(new_rows)-len(changed),"changed_input_inferred":len(changed),
                       "probabilities_sha256":digest_file(output/f"seed{seed}_probabilities.json")})
        members.append(probabilities)
        print(json.dumps({"fold":fold,**audits[-1]}),flush=True)
    means = {(row["pmc_id"],row["patient_id"],row["concept"]):sum(member[key] for member in members)/len(members) for key,row in new_index.items()}
    consumed,predictions = set(),[]
    for identifier in val_ids:
        scores=[]
        for row in context_scores[identifier]:
            key=identifier,row["patient_id"],row["concept"];consumed.add(key)
            scores.append({**row,"score":.75*row["score"]+.25*means[key]})
        predictions.append(from_scores(blind(docs[identifier]),entities[identifier],scores,.5))
    if consumed != set(means): raise ValueError("Filtered structural and text candidate universes differ.")
    write_jsonl(output/"predictions.jsonl",predictions)
    summary={"status":"completed","fold":fold,"plan_sha256":digest_file(plan_path),"validation_document_ids":val_ids,
             "concept_review":concept_audit,"members":audits,"relation_pairs":len(new_rows),
             "predictions_sha256":digest_file(output/"predictions.jsonl"),"elapsed_seconds":time.time()-started,
             "physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES")}
    write_json(output/"summary.json",summary)
    print(json.dumps(summary),flush=True)


def evaluate_joint(plan_path,plan):
    report=ROOT/plan["report_dir"]
    if (report/"summary.json").exists(): raise FileExistsError("Preserve completed joint evaluation.")
    docs,baseline,folds=inputs(plan)
    source,original,predictions,rows,audits=[],[],[],[],[]
    for fold in plan["pilot_folds"]:
        work=ROOT/plan["work_dir"]/f"fold{fold}"
        evidence=json.loads((work/"summary.json").read_text())
        if evidence["status"]!="completed" or evidence["plan_sha256"]!=digest_file(plan_path) or evidence["predictions_sha256"]!=digest_file(work/"predictions.jsonl"):
            raise ValueError("Joint inference is incomplete or changed.")
        ids=folds[fold];validation=[docs[i] for i in ids];base=[baseline[i] for i in ids];candidate=read_jsonl(work/"predictions.jsonl")
        old,metrics=evaluate(validation,base),evaluate(validation,candidate)
        row={"fold":fold,"baseline":old,"metrics":metrics,"delta":metrics["score"]-old["score"]};rows.append(row)
        source.extend(validation);original.extend(base);predictions.extend(candidate);audits.append(evidence)
        write_json(report/f"fold{fold}_metrics.json",row);write_json(report/f"fold{fold}_inference.json",evidence)
    old,metrics=evaluate(source,original),evaluate(source,predictions)
    deltas=[row["delta"] for row in rows];gates=plan["promotion"]
    checks={"minimum_mean_gain":sum(deltas)/len(deltas)>=gates["minimum_mean_gain"],"worst_fold":min(deltas)>=-gates["maximum_fold_loss"],
            "mention_improved":metrics["mention"]["f1"]>old["mention"]["f1"],
            "component_guard":all(metrics[k]["f1"]>=old[k]["f1"]-gates["maximum_component_loss"] for k in ["mention","document","association_micro","association_macro"]),
            "format_guard":all(a["concept_review"]["format_success_rate"]>=.98 for a in audits)}
    write_jsonl(report/"predictions.jsonl",predictions)
    validate_file(report/"predictions.jsonl",source,Ontology(ROOT/plan["ontology_path"]),report/"validation.json")
    result={"plan_sha256":digest_file(plan_path),"baseline":old,"metrics":metrics,"fold_deltas":deltas,"mean_gain":sum(deltas)/len(deltas),
            "checks":checks,"promoted":all(checks.values()),"bootstrap":bootstrap_delta(source,original,predictions),"errors":error_counts(source,predictions),
            "scope":"Single frozen joint pipeline on two reused development folds. Not additive component scores, independent validation or official B accuracy."}
    result["selected"]="joint_all_reviewed" if result["promoted"] else None
    result["ranked"]=[{"policy":"joint_all_reviewed","metrics":metrics,"fold_deltas":deltas,"mean_gain":result["mean_gain"],
                       "checks":checks,"eligible":result["promoted"],"errors":result["errors"]}]
    write_json(report/"plan.json",plan);write_json(report/"summary.json",result)
    print(json.dumps({k:result[k] for k in ["promoted","fold_deltas","mean_gain","checks"]}|{"score":metrics["score"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["infer","evaluate"])
    parser.add_argument("--plan",type=Path,required=True);parser.add_argument("--fold",type=int)
    args=parser.parse_args();plan=load_plan(args.plan)
    if args.stage=="infer":infer(args.plan.resolve(),plan,args.fold)
    else:evaluate_joint(args.plan.resolve(),plan)
