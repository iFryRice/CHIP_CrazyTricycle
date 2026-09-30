"""Stage the remaining pinned shards through the workstation's verified transport."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    request = json.loads((ROOT / "work/span_ner/qwen3_download/request.json").read_text())
    manifest = json.loads((ROOT / "experiments/llm_extraction/model.json").read_text())
    if request["manifest"] != manifest:
        raise ValueError("Transport request differs from the pinned model manifest.")
    destination = ROOT / "work/span_ner/qwen3_download/transport"
    destination.mkdir(exist_ok=True)
    for name in sorted(request["weights_download_urls"]):
        if name == "model-00001-of-00005.safetensors":
            continue
        output = destination / name
        if output.parent != destination:
            raise ValueError("Transport path escapes its staging directory.")
        sys.argv = ["download_verified_file.py", "--url", request["weights_download_urls"][name],
                    "--sha256", manifest["files"][name]["sha256"], "--output", str(output), "--workers", "16"]
        runpy.run_path(str(ROOT / "scripts/download_verified_file.py"), run_name="__main__")
        subprocess.run(["scp", "-o", "BatchMode=yes", str(output),
                        f"10.253.27.177:/home/dcf/chip2026/models/qwen3-8b/incoming/{name}"], check=True)
        print(json.dumps({"event": "transferred_to_incoming", "name": name, "bytes": output.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
