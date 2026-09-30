"""Verify and publish workstation-transferred weights under the pinned snapshot."""

import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    started = time.time()
    manifest_path = ROOT / "experiments/llm_extraction/model.json"
    spec = json.loads((ROOT / "experiments/llm_extraction/environment.json").read_text())
    if digest(manifest_path) != spec["model_manifest_sha256"]:
        raise ValueError("The pinned model manifest changed.")
    manifest = json.loads(manifest_path.read_text())
    destination = ROOT / "models/qwen3-8b"
    incoming = destination / "incoming"
    first = destination / "model-00001-of-00005.safetensors"
    if not first.exists():
        raise RuntimeError("The original server download has not finished its first shard.")
    verified = {}
    for name, info in sorted(manifest["files"].items()):
        path = destination / name
        if path.parent != destination:
            raise ValueError("Manifest path escapes the model directory.")
        if not path.exists():
            source = incoming / name
            while not source.exists() or source.stat().st_size < info["bytes"]:
                if time.time() - started > 5400:
                    raise TimeoutError("Workstation transfer did not finish within 90 minutes.")
                time.sleep(10)
            if source.stat().st_size != info["bytes"] or digest(source) != info["sha256"]:
                raise ValueError(f"Transferred model file differs from the pinned bytes: {name}")
            if path.exists():
                raise FileExistsError("Another process published the same model path.")
            source.rename(path)
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Published model file differs from the pinned bytes: {name}")
        verified[name] = info["sha256"]
        print(json.dumps({"event": "published_verified_file", "name": name, "bytes": info["bytes"]}), flush=True)
    result = {"status": "verified", "model_id": manifest["model_id"], "revision": manifest["revision"],
              "files": verified, "total_bytes": sum(entry["bytes"] for entry in manifest["files"].values()),
              "transport": "First shard from server; remaining shards from the same official snapshot through workstation and SSH.",
              "elapsed_seconds": time.time() - started}
    (ROOT / "reports/llm_extraction/model_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "model_verified", "total_bytes": result["total_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
