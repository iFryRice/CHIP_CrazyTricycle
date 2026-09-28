"""Verify pinned official model files before promoting a partial download."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    verified = []
    for entry in manifest["files"]:
        name = entry["name"]
        if Path(name).name != name:
            raise ValueError("Model manifest must contain simple file names")
        destination = args.model_dir / name
        source = destination if destination.exists() else destination.with_name(name + ".part")
        size = source.stat().st_size
        if size != entry["size"]:
            raise ValueError(f"Incomplete model file: {name}: {size} != {entry['size']}")
        sha256 = hashlib.sha256()
        git_hash = hashlib.sha1(f"blob {size}\0".encode())
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha256.update(block)
                git_hash.update(block)
        expected = entry.get("lfs_sha256")
        if expected:
            if sha256.hexdigest() != expected:
                raise ValueError(f"Official LFS SHA256 mismatch: {name}")
        elif entry.get("git_blob"):
            if git_hash.hexdigest() != entry["git_blob"]:
                raise ValueError(f"Official Git blob mismatch: {name}")
        else:
            raise ValueError(f"No official digest available: {name}")
        if source != destination:
            source.rename(destination)
        verified.append({"name": name, "size": size, "sha256": sha256.hexdigest()})
    result = {"verified": True, "model": manifest["model"],
              "revision": manifest["revision"], "files": verified}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
