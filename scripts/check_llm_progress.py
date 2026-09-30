"""Print compact read-only status for the model, acceptance and frozen pilot."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/llm_extraction"


def read_status(name):
    path = REPORT / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"status": "report_write_in_progress"}


def tail(path, maximum_bytes=1200):
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - maximum_bytes))
        return handle.read().decode("utf-8", errors="replace")


def main():
    manifest = json.loads((ROOT / "experiments/llm_extraction/model.json").read_text())
    plan = json.loads((ROOT / "experiments/llm_extraction/plan.json").read_text())
    directory = ROOT / "models/qwen3-8b"
    weights = []
    for name, item in manifest["files"].items():
        if not name.endswith(".safetensors"):
            continue
        path, incoming = directory / name, directory / "incoming" / name
        weights.append({"file": name, "expected_bytes": item["bytes"],
                        "published_bytes": path.stat().st_size if path.exists() else 0,
                        "incoming_bytes": incoming.stat().st_size if incoming.exists() else 0})
    queue = read_status("pilot_queue.json")
    jobs = []
    for fold in [0, 1]:
        folder = ROOT / "work/span_ner/llm_extraction" / f"fold{fold}"
        caches = sorted((folder / "prompts").glob("*.json"))
        recent = [json.loads(path.read_text())["elapsed_seconds"] for path in caches[-50:]]
        expected = plan["expected_chunks"][str(fold)]
        jobs.append({"fold": fold, "cached_prompts": len(caches), "expected_prompts": expected,
                     "recent_mean_generation_seconds": sum(recent) / len(recent) if recent else None,
                     "estimated_remaining_seconds": (expected-len(caches)) * sum(recent) / len(recent) if recent else None,
                     "summary_exists": (folder / "summary.json").exists(),
                     "log_tail": tail(REPORT / f"fold{fold}_inference.log", 500)})
    processes = []
    output = subprocess.run(["ps", "-eo", "pid,etime,args"], check=True, capture_output=True, text=True).stdout
    names = ["finalize_qwen_stage.py", "qwen_runtime_witness.py", "run_llm_pilot_queue.py", "infer_llm_extraction.py", "evaluate_llm_extraction.py"]
    for line in output.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and fields[2].split()[0].endswith("python"):
            matches = [name for name in names if f"scripts/{name}" in fields[2]]
            if matches:
                processes.append({"pid": int(fields[0]), "elapsed": fields[1], "script": matches[0]})
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True)
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "processes": processes, "weights": weights,
              "model_verification": (read_status("model_verification.json") or {}).get("status"),
              "smoke": (read_status("environment_smoke.json") or {}).get("status"),
              "independent_acceptance": (read_status("agent_acceptance.json") or {}).get("status"),
              "queue": queue, "jobs": jobs, "gpu": gpu.stdout.strip().splitlines(),
              "smoke_log_tail": tail(REPORT / "environment_smoke.log"),
              "evaluation_summary": read_status("pilot/summary.json")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
