"""Download one pinned large file with strict HTTP ranges and a final SHA256."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{64}", args.sha256):
        raise ValueError("An exact expected SHA256 is required")
    if args.output.exists():
        if digest(args.output) != args.sha256:
            raise ValueError("Existing output has a different digest; refusing overwrite")
        print("Already verified", flush=True)
        return
    with urlopen(Request(args.url, method="HEAD"), timeout=45) as response:
        content_range = response.headers.get("Content-Range")
        size = int(content_range.rsplit("/", 1)[1] if content_range
                   else response.headers["Content-Length"])
    if size < 1:
        raise ValueError("Invalid advertised file size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts = args.output.with_name(args.output.name + ".chunks")
    parts.mkdir(exist_ok=True)
    identity = {"url": args.url, "sha256": args.sha256, "size": size}
    identity_file = parts / "identity.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise ValueError("Partial chunks belong to a different file")
    else:
        identity_file.write_text(json.dumps(identity), encoding="utf-8")
    chunk_size = 4 * 1024 * 1024
    ranges = [(start, min(size, start + chunk_size) - 1)
              for start in range(0, size, chunk_size)]

    def fetch(index):
        start, end = ranges[index]
        path = parts / f"{index:05d}.part"
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            return expected
        for attempt in range(4):
            try:
                request = Request(args.url, headers={"Range": f"bytes={start}-{end}",
                                                     "Accept-Encoding": "identity"})
                with urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Server did not honor the exact requested byte range")
                    received = 0
                    with path.open("wb") as stream:
                        while block := response.read(min(1024 * 1024, expected - received + 1)):
                            received += len(block)
                            if received > expected:
                                raise ValueError("Range response exceeded its declared size")
                            stream.write(block)
                    if received != expected:
                        raise ValueError("Incomplete range response")
                return expected
            except Exception:
                if path.exists():
                    path.unlink()
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("Unreachable range state")

    completed = 0
    print(json.dumps({"event": "start", "bytes": size, "parts": len(ranges),
                      "workers": args.workers}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, index) for index in range(len(ranges))]
        for future in as_completed(futures):
            completed += future.result()
            print(json.dumps({"event": "progress", "bytes": completed, "total": size}), flush=True)
    combined = args.output.with_name(args.output.name + ".assembling")
    with combined.open("wb") as destination:
        for index in range(len(ranges)):
            with (parts / f"{index:05d}.part").open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(block)
    actual = digest(combined)
    if actual != args.sha256:
        raise ValueError(f"Final SHA256 mismatch: {actual}")
    combined.rename(args.output)
    for index in range(len(ranges)):
        (parts / f"{index:05d}.part").unlink()
    identity_file.unlink()
    parts.rmdir()
    print(json.dumps({"event": "verified", "bytes": size, "sha256": actual}), flush=True)


if __name__ == "__main__":
    main()
