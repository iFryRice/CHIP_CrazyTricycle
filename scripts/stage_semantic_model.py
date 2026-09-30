"""Stage a pinned SapBERT artifact using the existing verified range downloader."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    manifest = request["manifest"]
    if manifest["model_id"] != "cambridgeltl/SapBERT-from-PubMedBERT-fulltext":
        raise ValueError("Unexpected model identity.")
    destination = ROOT / "models/sapbert"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, contents in request["small_files"].items():
        path = destination / filename
        if path.parent != destination:
            raise ValueError("Model file escapes destination.")
        data = base64.b64decode(contents)
        if hashlib.sha256(data).hexdigest() != manifest["files"][filename]["sha256"]:
            raise ValueError("Small model file failed its content hash.")
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError("Refusing to replace a different model file.")
        path.write_bytes(data)
    weights = manifest["files"]["model.safetensors"]
    original_argv = sys.argv
    try:
        sys.argv = ["download_verified_file.py", "--url", request["weights_download_url"],
                    "--sha256", weights["sha256"], "--output", str(destination / "model.safetensors"), "--workers", "8"]
        runpy.run_path(str(ROOT / "scripts/download_verified_file.py"), run_name="__main__")
    finally:
        sys.argv = original_argv
    verified = {}
    for filename, expected in manifest["files"].items():
        path = destination / filename
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        if value != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Model file failed final verification: {filename}")
        verified[filename] = value
    record = {"status": "verified", "model_id": manifest["model_id"], "revision": manifest["revision"], "files": verified}
    report = ROOT / "reports/semantic_linking/model_verification.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
