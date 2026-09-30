"""Generate a blind B candidate only after frozen fivefold confirmation passes."""

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.ontology import Ontology
from patientphex.patient_linking import _check_provenance
from patientphex.relation_text import Collator, build_model, configure_tokenizer, encode, examples
from scripts.optimize_cpu_association import blind, from_scores, validate_file
from scripts.run_llm_association import atomic_json, load_plan
from scripts.run_joint_review import filtered_entities
from scripts.run_joint_b import check_confirmation


def run(plan_path):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import transformers

    plan = load_plan(plan_path)
    confirmation_plan = check_confirmation(plan)
    ensemble_plan = load_plan(ROOT / plan["ensemble_plan"])
    confirmation = json.loads((ROOT / plan["confirmation_summary"]).read_text())
    if confirmation["promoted"] is not True or not all(confirmation["checks"].values()):
        raise ValueError("Fivefold confirmation did not pass; B generation is not allowed.")
    if confirmation["selected"] != "joint_all_reviewed":
        raise ValueError("The confirmed policy differs from the B procedure.")
    if ensemble_plan["seeds"] != plan["seeds"] or confirmation_plan["fixed"]["seeds"] != plan["seeds"]:
        raise ValueError("The declared seed ensemble differs from confirmation.")
    members = plan["text_models"]
    if (len(members) != 5*len(plan["seeds"])
            or {(item["seed"],item["fold"]) for item in members.values()} != {(seed,fold) for seed in plan["seeds"] for fold in range(5)}):
        raise ValueError("B inference must include every predeclared seed and fold exactly once.")
    if (not torch.cuda.is_available() or torch.__version__ != "2.6.0+cu124"
            or transformers.__version__ != "4.49.0"):
        raise RuntimeError("The accepted inference runtime changed.")
    work, report, output = [ROOT / plan[key] for key in ["work_dir", "report_dir", "output"]]
    if work.exists() or output.exists() or (report / "summary.json").exists():
        raise FileExistsError("Preserve previously generated candidates and inference results.")
    started = time.time()
    work.mkdir(parents=True)
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "plan.json", plan)

    def progress(phase, **values):
        event = {"phase": phase, "pid": os.getpid(), "plan_sha256": digest_file(plan_path),
                 "elapsed_seconds": time.time()-started, **values}
        atomic_json(report / "progress.json", event)
        print(json.dumps(event), flush=True)

    progress("preparing")
    training = read_jsonl(ROOT / plan["train_path"])
    target = read_jsonl(ROOT / plan["target_path"])
    target_ids = {str(d["pmc_id"]) for d in target}
    training_ids = {str(d["pmc_id"]) for d in training}
    if (len(target) != 100 or len(target_ids) != 100 or sum(len(d["patient"]) for d in target) != 244
            or target_ids & training_ids or any(d.get("entities") or d.get("association") for d in target)):
        raise ValueError("Expected the complete, disjoint, blind 100-document B set.")
    baseline = read_jsonl(ROOT / plan["baseline_b"])
    indexed = {str(d["pmc_id"]): d for d in baseline}
    if len(indexed) != len(baseline) or set(indexed) != target_ids:
        raise ValueError("The baseline B candidate does not cover the complete source.")
    with (ROOT / plan["association_model"]).open("rb") as handle:
        structural = pickle.load(handle)
    _check_provenance(structural.provenance_, training_ids, target_ids)
    structural_scores, reproduced = {}, []
    for document in target:
        identifier = str(document["pmc_id"])
        old = structural.predict_scores(blind(document), indexed[identifier]["entities"])
        structural_scores[identifier] = old
        reproduced.append(from_scores(document, indexed[identifier]["entities"], old, 0.5))
    if reproduced != baseline:
        raise ValueError("The existing B baseline failed exact structural replay.")
    ontology = Ontology(ROOT / plan["ontology_path"])
    concept_path = ROOT / plan["concept_plan"]
    concept_plan = load_plan(concept_path)
    original_concept = load_plan(ROOT / confirmation_plan["concept_confirmation_plan"])
    for name in ["protocol_module", "generation", "fixed", "identity_fields", "environment_spec"]:
        if concept_plan[name] != original_concept[name]:
            raise ValueError("Blind B concept review differs from the confirmed protocol.")
    entities, concept_audits = {}, []
    for shard in concept_plan["pilot_folds"]:
        manifest = json.loads((ROOT / concept_plan["work_dir"] / f"fold{shard}" / "prepared.json").read_text())
        identifiers = manifest["validation_document_ids"]
        if (manifest.get("partition_role") != "blind_b_shard"
                or set(identifiers) & set(entities) or not set(identifiers) <= target_ids
                or set(manifest["upstream_training_document_ids"]) != training_ids):
            raise ValueError("B concept shards overlap or have invalid source provenance.")
        filtered, audit = filtered_entities(
            {"concept_plans": {str(shard): plan["concept_plan"]}}, shard, indexed, identifiers, ontology)
        if audit["format_success_rate"] < 0.98:
            raise ValueError("B concept review failed the frozen format gate.")
        entities.update(filtered)
        concept_audits.append(audit)
    if set(entities) != target_ids:
        raise ValueError("B concept review does not cover all source documents.")
    structural_scores = {str(d["pmc_id"]): structural.predict_scores(blind(d), entities[str(d["pmc_id"])]) for d in target}
    write_json(work / "filtered_entities.json", entities)
    write_json(work / "context_scores.json", structural_scores)
    model_path = ROOT / plan["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    configure_tokenizer(tokenizer)
    rows = [encode(tokenizer, row) for d in target for row in examples(blind(d), entities[str(d["pmc_id"])], ontology)]
    keys = [(row["pmc_id"], row["patient_id"], row["concept"]) for row in rows]
    expected = {(identifier, row["patient_id"], row["concept"]) for identifier, scores in structural_scores.items() for row in scores}
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("B text and structural candidate universes differ.")
    write_json(work / "example_identity.json", [{key: row[key] for key in ["task_id", "pmc_id", "patient_id", "concept"]} for row in rows])
    totals = [0.0]*len(rows)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    artifacts = {}
    for member_id, item in members.items():
        fold, seed = item["fold"], item["seed"]
        manifest = json.loads((ROOT / item["manifest"]).read_text())
        val_ids = json.loads((ROOT / plan["split_path"]).read_text())["folds"][fold]
        if (set(manifest["training_document_ids"]) != training_ids-set(val_ids)
                or manifest["validation_document_ids"] != val_ids or manifest["smoke"]
                or manifest["parameters"] != {**ensemble_plan["parameters"], "seed": seed}):
            raise ValueError("A B ensemble member has unexpected training provenance.")
        checkpoint = torch.load(ROOT / item["checkpoint"], map_location="cpu", weights_only=True)
        if checkpoint["smoke"] or checkpoint["epochs_completed"] != 3 or checkpoint["fold"] != fold or checkpoint["plan_sha256"] != item["training_plan_sha256"]:
            raise ValueError("An ensemble checkpoint differs from its confirmed training run.")
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        encoder = AutoModel.from_config(config, trust_remote_code=False, attn_implementation="eager")
        encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model = build_model(encoder)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        model.cuda().eval()
        probabilities = []
        with torch.inference_mode():
            for batch in DataLoader(rows, batch_size=24, shuffle=False, collate_fn=Collator(tokenizer)):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    values = model(**{name: value.cuda() for name,value in batch["inputs"].items()}).float().sigmoid()
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Nonfinite B probability.")
                probabilities.extend(values.cpu().tolist())
                if len(probabilities) % 600 == 0:
                    progress("inference", member=member_id, completed=len(probabilities), total=len(rows))
        if len(probabilities) != len(rows):
            raise ValueError("Incomplete ensemble inference.")
        write_json(work / f"{member_id}_probabilities.json", probabilities)
        artifacts[member_id] = digest_file(work / f"{member_id}_probabilities.json")
        totals = [total+probability/len(members) for total,probability in zip(totals, probabilities, strict=True)]
        del model, encoder
        torch.cuda.empty_cache()
        progress("member_completed", member=member_id, completed_members=len(artifacts), total_members=len(members), pairs=len(rows))
    neural = dict(zip(keys, totals, strict=True))
    predictions = []
    added = removed = changed_patients = 0
    for document in target:
        identifier = str(document["pmc_id"])
        old = indexed[identifier]
        scores = [{**row, "score": 0.75*row["score"]+0.25*neural[identifier,row["patient_id"],row["concept"]]}
                  for row in structural_scores[identifier]]
        prediction = from_scores(document, entities[identifier], scores, 0.5)
        if prediction["entities"] != entities[identifier]:
            raise ValueError("The association stage changed the reviewed entity records.")
        previous = {a["patient_id"]: set(a["phenotype"]) for a in old["association"]}
        for association in prediction["association"]:
            before, after = previous[association["patient_id"]], set(association["phenotype"])
            added += len(after-before); removed += len(before-after); changed_patients += before != after
        predictions.append(prediction)
    candidate = work / "candidate.jsonl"
    write_jsonl(candidate, predictions)
    validated = validate_file(candidate, target, ontology, report / "validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, output)
    if digest_file(output) != validated["sha256"]:
        raise ValueError("Published candidate differs from the validated artifact.")
    result = {"status": "completed", "plan_sha256": digest_file(plan_path), "output": plan["output"],
              "sha256": digest_file(output), "validation": validated, "source_pairs": len(rows),
              "text_fold_probability_sha256": artifacts, "model_count": len(members),
              "association_added": added, "association_removed": removed, "changed_patients": changed_patients,
              "concept_review": concept_audits, "removed_entities": sum(a["removed_entities"] for a in concept_audits),
              "retained_entity_records_exactly_preserved": True, "elapsed_seconds": time.time()-started,
              "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
              "official_score": None, "scope": "Blind B inference with all predeclared folds and seeds. Development uses each document's held-out seed ensemble; ensemble B accuracy requires official feedback."}
    write_json(report / "summary.json", result)
    (report / "report.md").write_text("\n".join(["# B 集概念核对与关联融合候选", "",
        f"文件：`{plan['output']}`；SHA256：`{result['sha256']}`。", "",
        f"确认全五折预设门槛通过后生成。按固定 Qwen 规则删除 {result['removed_entities']} 条概念含义不符实体，保留其余实体的 HPO 映射及原文偏移；保护否定、未映射、复合 HPO 和唯一精确官方词条匹配。过滤后重新计算关联概率：75% 全量训练上下文模型 + 25% 全部 {len(members)} 个文本模型的等权平均，阈值 0.5。", "",
        f"完整覆盖 100 篇文献、244 位患者；关联增加 {added} 条，减少 {removed} 条，影响 {changed_patients} 位患者。严格格式和偏移校验通过。", "",
        "这是一份待提交候选，尚无官方分数。全五折开发使用各文献自己的折外种子集成，B 使用全部折与种子，不能将开发分数当作 B 集成的已测性能。", "",
        "原纯 CPU 官方分数仍为 0.6152。原 B 上下文候选保留，未向比赛平台提交。", ""]), encoding="utf-8")
    progress("completed", output=plan["output"], sha256=result["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    run(parser.parse_args().plan.resolve())
