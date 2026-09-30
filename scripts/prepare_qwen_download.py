"""Resolve a pinned official Qwen snapshot without printing signed URLs."""

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_ID = "Qwen/Qwen3-8B"


def main():
    with urlopen(f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true", timeout=45) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION or metadata.get("cardData", {}).get("license") != "apache-2.0":
        raise ValueError("Official model revision/license changed.")
    keep = {"config.json", "generation_config.json", "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json", "LICENSE"}
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "license": "apache-2.0",
                "parameter_count": metadata["safetensors"]["total"], "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    if manifest["parameter_count"] > 10_000_000_000:
        raise ValueError("The model exceeds the competition parameter limit.")
    small_files, weights_urls = {}, {}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        weights = name.endswith(".safetensors")
        if not weights and name not in keep:
            continue
        if Path(name).name != name:
            raise ValueError("Unexpected nested model path.")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        if weights:
            with urlopen(Request(url, method="HEAD", headers={"Cache-Control": "no-cache"}), timeout=45) as response:
                resolved_url = response.url
            if urlsplit(resolved_url).scheme != "https":
                raise ValueError("Official redirect downgraded HTTPS.")
            weights_urls[name] = resolved_url
            value = item["lfs"]["sha256"]
        else:
            with urlopen(url, timeout=60) as response:
                data = response.read()
            if len(data) != item["size"]:
                raise ValueError("Small file length mismatch.")
            value = hashlib.sha256(data).hexdigest()
            if item.get("lfs"):
                if value != item["lfs"]["sha256"]:
                    raise ValueError("Small LFS file SHA256 mismatch.")
            elif hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != item["blobId"]:
                raise ValueError("Git blob identity mismatch.")
            small_files[name] = base64.b64encode(data).decode()
        manifest["files"][name] = {"sha256": value, "bytes": item["size"], "git_blob": item["blobId"]}
        print(json.dumps({"prepared": name, "bytes": item["size"]}), flush=True)
    destination = ROOT / "experiments/llm_extraction/model.json"
    if destination.exists():
        raise FileExistsError("Refusing to replace the declared model snapshot.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    request = ROOT / "work/span_ner/qwen3_download/request.json"
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(json.dumps({"manifest": manifest, "small_files": small_files, "weights_download_urls": weights_urls}), encoding="utf-8")
    print(json.dumps({"model": MODEL_ID, "revision": REVISION, "parameter_count": manifest["parameter_count"],
                      "total_bytes": sum(entry["bytes"] for entry in manifest["files"].values())}), flush=True)


if __name__ == "__main__":
    main()
