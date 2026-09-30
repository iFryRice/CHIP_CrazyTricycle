"""Stage the exact Qwen3 snapshot using the verified range downloader."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError("Download concurrency must lie in [1, 64].")
    request = json.loads(args.request.read_text())
    manifest = json.loads((ROOT / "experiments/llm_extraction/model.json").read_text())
    if manifest != request["manifest"] or manifest["model_id"] != "Qwen/Qwen3-8B" or manifest["revision"] != "b968826d9c46dd6066d109eabc6255188de91218":
        raise ValueError("Download request does not match the declared model.")
    destination = ROOT / "models/qwen3-8b"
    destination.mkdir(parents=True, exist_ok=True)
    for name, encoded in request["small_files"].items():
        path = destination / name
        if path.parent != destination:
            raise ValueError("Model path escapes its destination.")
        data = base64.b64decode(encoded)
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError("Small model file hash mismatch.")
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError("Refusing to overwrite a different model file.")
        path.write_bytes(data)
    for name, url in sorted(request["weights_download_urls"].items()):
        path = destination / name
        if path.parent != destination or not name.endswith(".safetensors"):
            raise ValueError("Unexpected weight path.")
        info = manifest["files"][name]
        if not path.exists() and shutil.disk_usage(destination).free < info["bytes"] * 2 + 4 * 1024 ** 3:
            raise RuntimeError("Insufficient disk space for verified range assembly.")
        print(json.dumps({"event": "shard_start", "name": name, "bytes": info["bytes"]}), flush=True)
        previous = sys.argv
        try:
            sys.argv = ["download_verified_file.py", "--url", url, "--sha256", info["sha256"], "--output", str(path), "--workers", str(args.workers)]
            runpy.run_path(str(ROOT / "scripts/download_verified_file.py"), run_name="__main__")
        finally:
            sys.argv = previous
    verified = {}
    for name, info in manifest["files"].items():
        path = destination / name
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Final model verification failed: {name}")
        verified[name] = info["sha256"]
    report = ROOT / "reports/llm_extraction/model_verification.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "verified", "model_id": manifest["model_id"], "revision": manifest["revision"], "files": verified,
              "total_bytes": sum(entry["bytes"] for entry in manifest["files"].values())}
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
