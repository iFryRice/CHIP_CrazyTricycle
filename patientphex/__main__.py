"""Command-line entry point for reproducible local CPU experiments."""

import argparse
import importlib.metadata
import platform
import statistics
import time
from pathlib import Path

from .data import (
    digest_file, document_folds, prediction_record, read_jsonl, write_json, write_jsonl,
)


def blind(document):
    """Remove all answer fields before every prediction, including training inputs."""
    return {**document, "entities": [], "association": []}


def log(message):
    print(message, flush=True)


def generate_predictions(documents, entities, model):
    return [
        prediction_record(doc, spans, model.predict(blind(doc), spans))
        for doc, spans in zip(documents, entities, strict=True)
    ]


def run(args):
    from .association import AssociationModel
    from .entities import EntityExtractor
    from .evaluation import evaluate
    from .ontology import Ontology
    from .validation import validate_submission

    started = time.monotonic()
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    train_path = data_dir / "PatientPheX-train.jsonl"
    target_path = data_dir / "PatientPheX-A.jsonl"
    ontology_path = data_dir / "hp.obo"
    original_hashes = {str(path): digest_file(path) for path in (train_path, target_path, ontology_path)}
    code_paths = sorted(Path(__file__).parent.glob("*.py")) + [Path("pyproject.toml"), Path("uv.lock")]
    code_hashes = {str(path): digest_file(path) for path in code_paths}
    documents = read_jsonl(train_path)
    targets = read_jsonl(target_path)
    if set(str(d["pmc_id"]) for d in documents) & set(str(d["pmc_id"]) for d in targets):
        raise ValueError("Training and target documents overlap")
    if any(d["entities"] or d["association"] for d in targets):
        raise ValueError("Expected an unlabeled A set")
    ontology = Ontology(ontology_path)
    if "2026-06-23" not in str(ontology.version):
        raise ValueError(f"Unexpected HPO version: {ontology.version}")
    fold_documents = document_folds(documents, args.folds, args.seed)
    split = {
        "seed": args.seed,
        "strategy": "Document-level, stratified by one versus multiple target patients; SHA256 ordering.",
        "folds": [[d["pmc_id"] for d in fold] for fold in fold_documents],
    }
    write_json(report_dir / "split.json", split)
    combinations = [(e, a) for e in ("dictionary", "learned") for a in ("nearest", "learned")]
    accumulated = {f"{e}+{a}": [] for e, a in combinations}
    fold_scores = {name: [] for name in accumulated}
    log(f"Documents: train={len(documents)}, A={len(targets)}; HPO terms={len(ontology.allowed_ids)}")
    for fold_index, held_out in enumerate(fold_documents):
        held_out_ids = {str(d["pmc_id"]) for d in held_out}
        training = [d for d in documents if str(d["pmc_id"]) not in held_out_ids]
        log(f"Fold {fold_index + 1}/{args.folds}: fit {len(training)}, validate {len(held_out)}")
        extractor = EntityExtractor(ontology, mode="learned").fit(training)
        for entity_mode in ("dictionary", "learned"):
            extractor.mode = entity_mode
            training_entities = [extractor.predict(blind(d)) for d in training]
            validation_entities = [extractor.predict(blind(d)) for d in held_out]
            for association_mode in ("nearest", "learned"):
                name = f"{entity_mode}+{association_mode}"
                model = AssociationModel(mode=association_mode).fit(training, training_entities)
                predictions = generate_predictions(held_out, validation_entities, model)
                result = evaluate(held_out, predictions)
                accumulated[name].extend(predictions)
                fold_scores[name].append(result)
                log(f"  {name}: score={result['score']:.4f}")

    comparisons = {}
    for name, predictions in accumulated.items():
        scores = [m["score"] for m in fold_scores[name]]
        comparisons[name] = {
            "pooled_out_of_fold": evaluate(documents, predictions),
            "fold_score_mean": statistics.mean(scores),
            "fold_score_std": statistics.pstdev(scores),
            "folds": fold_scores[name],
        }
    selected = max(comparisons, key=lambda name: comparisons[name]["pooled_out_of_fold"]["score"])
    selected_metrics = comparisons[selected]["pooled_out_of_fold"]
    oof_path = report_dir / "out_of_fold_predictions.jsonl"
    prediction_map = {str(d["pmc_id"]): d for d in accumulated[selected]}
    ordered_oof = [prediction_map[str(d["pmc_id"])] for d in documents]
    write_jsonl(oof_path, ordered_oof)
    oof_validation = validate_submission(oof_path, documents, ontology)
    log(f"Selected {selected}; pooled out-of-fold score={selected_metrics['score']:.4f}")

    entity_mode, association_mode = selected.split("+")
    extractor = EntityExtractor(ontology, mode=entity_mode).fit(documents)
    training_entities = [extractor.predict(blind(d)) for d in documents]
    model = AssociationModel(mode=association_mode).fit(documents, training_entities)
    target_entities = [extractor.predict(blind(d)) for d in targets]
    predictions = generate_predictions(targets, target_entities, model)
    write_jsonl(args.output, predictions)
    submission_validation = validate_submission(args.output, targets, ontology)
    if any(digest_file(path) != checksum for path, checksum in original_hashes.items()):
        raise RuntimeError("An original input changed during execution")
    if any(digest_file(path) != checksum for path, checksum in code_hashes.items()):
        raise RuntimeError("Implementation changed during execution; rerun with stable source files")
    metrics = {
        "selected_configuration": selected,
        "selection_rule": "Highest pooled document-held-out score among four predeclared configurations.",
        "score_caveat": "Cross-validation used for model selection, not an independent test score. Local scorer approximates public rules; A leaderboard score is unknown.",
        "selected_metrics": selected_metrics,
        "comparisons": comparisons,
        "input_sha256": original_hashes,
        "code_sha256": code_hashes,
        "submission_sha256": digest_file(args.output),
        "environment": {"python": platform.python_version(), "scikit_learn": importlib.metadata.version("scikit-learn")},
        "hpo_version": ontology.version,
        "runtime_seconds": round(time.monotonic() - started, 2),
        "training_documents": len(documents),
        "fold_count": args.folds,
        "target_documents": len(targets),
        "target_patients": sum(len(d["patient"]) for d in targets),
        "predicted_entities": sum(len(d["entities"]) for d in predictions),
        "predicted_associations": sum(len(a["phenotype"]) for d in predictions for a in d["association"]),
        "submission_bytes": Path(args.output).stat().st_size,
    }
    write_json(report_dir / "metrics.json", metrics)
    write_json(report_dir / "submission_validation.json", submission_validation)
    write_json(report_dir / "oof_validation.json", oof_validation)
    per_document = [
        {"pmc_id": d["pmc_id"], "patients": len(d["patient"]), "metrics": evaluate([d], [prediction_map[str(d["pmc_id"])]])}
        for d in documents
    ]
    write_json(report_dir / "per_document_metrics.json", per_document)
    write_report(report_dir / "report.md", metrics)
    log(f"Saved {args.output}: {metrics['submission_bytes']} bytes; validation passed.")


def write_report(path, metrics):
    score = metrics["selected_metrics"]
    lines = [
        "# PatientPheX CPU 基线报告", "",
        f"选择方案：`{metrics['selected_configuration']}`。训练文献 {metrics['training_documents']} 篇，A 榜文献 {metrics['target_documents']} 篇、患者 {metrics['target_patients']} 人。", "",
        "## 离线验证", "",
        f"按文献进行固定 {metrics['fold_count']} 折划分，单患者/多患者分层；每折词典、消歧统计和分类器仅使用该折训练文献。预测前移除答案字段。固定 HPO 本体可用于全部折。", "",
        "以下是用于选择方案的交叉验证分数，不能视为独立测试或 A 榜成绩。官方评测脚本未提供，本地实现依据项目内公开说明，存在正文范围及特殊实体处理差异。", "",
        "| 指标 | Precision | Recall | F1 |", "|---|---:|---:|---:|",
    ]
    for key, label in (("mention", "实体提及"), ("document", "文档概念"), ("association_micro", "患者关联 Micro"), ("association_macro", "患者关联 Macro")):
        item = score[key]
        lines.append(f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |")
    lines.extend(["", f"四项 F1 等权总分：**{score['score']:.4f}**。", "", "| 配置 | 折外汇总分 | 各折均值 ± 标准差 |", "|---|---:|---:|"])
    for name, comparison in metrics["comparisons"].items():
        lines.append(f"| {name} | {comparison['pooled_out_of_fold']['score']:.4f} | {comparison['fold_score_mean']:.4f} ± {comparison['fold_score_std']:.4f} |")
    lines.extend([
        "", "## 正式预测与校验", "",
        f"选定配置用全部 80 篇训练文献重建后预测 A 榜。输出 {metrics['predicted_entities']} 个实体、{metrics['predicted_associations']} 条患者概念关联；文件 {metrics['submission_bytes']:,} 字节，低于 100,000,000 字节。",
        "", "已检查完整文献覆盖、主键唯一、患者引用、原文字符跨度、HPO 分支、JSONL 编码与文件大小。输入文件 SHA256 和环境版本见 `metrics.json`；逐项检查见 `submission_validation.json`。",
        "", "## 适用范围", "",
        "本版仅用 CPU、训练标注和随数据提供的 HPO 本体，无外部大模型推理，无平台提交。主要局限是未见表达的召回、精确边界、词义歧义及跨段患者共指。",
        "", "## 复现", "", "```powershell", "uv sync --locked", "uv run python -m unittest discover -s tests -v", "uv run python -m patientphex run", "uv run python -m patientphex validate --input submissions/patientphex_a.jsonl", "```", "",
        "方法实现参考：[scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)；隔离训练与验证参照 [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)。", "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    runner = subparsers.add_parser("run", help="Cross-validate, refit, predict A, and validate")
    runner.add_argument("--data-dir", default="PatientPheX-V1-A")
    runner.add_argument("--report-dir", default="reports/cpu_baseline")
    runner.add_argument("--output", default="submissions/patientphex_a.jsonl")
    runner.add_argument("--folds", type=int, default=5)
    runner.add_argument("--seed", type=int, default=20260927)
    validator = subparsers.add_parser("validate", help="Validate an existing A-set submission")
    validator.add_argument("--input", required=True)
    validator.add_argument("--data-dir", default="PatientPheX-V1-A")
    validator.add_argument("--report", default="reports/cpu_baseline/submission_validation.json")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        from .ontology import Ontology
        from .validation import validate_submission
        result = validate_submission(args.input, read_jsonl(Path(args.data_dir) / "PatientPheX-A.jsonl"), Ontology(Path(args.data_dir) / "hp.obo"))
        write_json(args.report, result)
        log(str(result))


if __name__ == "__main__":
    main()
