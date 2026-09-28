"""Stream a bounded, HPO-restricted crosswalk from an existing read-only UMLS DB.

Run with /usr/bin/python3 on server 177; its system Python provides SQLite.
Only a small HPO-to-CUI dictionary is held in memory. Source terminology rows
are queried in indexed CUI batches and streamed directly to a gzip TSV cache.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCES = ("RXNORM", "LNC", "SNOMEDCT_US", "ICD9CM", "ICD10CM", "ICD10PCS", "MSH", "HPO", "MED-RT", "MTH")
FIELDS = ("term", "hpo_id", "cui", "sab", "source_code", "aui", "tty", "is_pref")


def allowed_hpo_ids(path: Path) -> tuple:
    """Read only the OBO branch fields using the SQLite-capable system Python.

    The server's system Python predates PEP 604, whereas the project Ontology
    module evaluates union annotations at import. This standalone parser avoids
    importing that module or changing either Python environment.
    """
    version = None
    terms = {}
    current = None
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith("data-version: "):
                version = line.rsplit("/", 1)[-1]
            if line.startswith("["):
                if current and current.get("id"):
                    terms[current["id"]] = current
                current = {"parents": [], "obsolete": False} if line == "[Term]" else None
            elif current is not None:
                if line.startswith("id: "):
                    current["id"] = line[4:]
                elif line.startswith("is_a: "):
                    current["parents"].append(line[6:].split()[0])
                elif line == "is_obsolete: true":
                    current["obsolete"] = True
    if current and current.get("id"):
        terms[current["id"]] = current
    children = defaultdict(set)
    for identifier, term in terms.items():
        if not term["obsolete"]:
            for parent in term["parents"]:
                children[parent].add(identifier)
    allowed = set()
    queue = deque(children["HP:0000118"])
    while queue:
        identifier = queue.popleft()
        if identifier not in allowed:
            allowed.add(identifier)
            queue.extend(children[identifier])
    return version, allowed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_terms(database: Path, ontology_path: Path, output: Path, *, batch_size: int = 200,
                 max_rows: int = 1000000) -> dict:
    if batch_size < 1 or batch_size > 500 or max_rows < 1:
        raise ValueError("batch_size must be 1..500 and max_rows must be positive")
    database, ontology_path, output = database.resolve(), ontology_path.resolve(), output.resolve()
    manifest_path = output.with_name(output.name + ".manifest.json")
    partial = output.with_name(output.name + ".partial")
    if output in {database, ontology_path} or any(p.exists() for p in (output, manifest_path, partial)):
        raise FileExistsError("Refusing to overwrite a source, completed export, or partial export")
    hpo_version, allowed_ids = allowed_hpo_ids(ontology_path)
    if hpo_version != "2026-06-23" or "HP:0000118" in allowed_ids:
        raise ValueError("Expected HPO 2026-06-23 descendants excluding the root")
    before = database.stat()
    db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    db.execute("BEGIN")
    cui_to_hpo = defaultdict(set)
    source_versions = db.execute("SELECT vsab, source_version FROM sources WHERE rsab='HPO'").fetchall()
    for code, cui in db.execute("SELECT DISTINCT code,cui FROM terms WHERE sab='HPO' AND lat='ENG' AND suppress='N' ORDER BY code,cui"):
        if code in allowed_ids:
            cui_to_hpo[cui].add(code)
    if not cui_to_hpo:
        db.close()
        raise ValueError("No current allowed HPO IDs were found in the UMLS source index")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    source_counts = Counter()
    covered = set().union(*cui_to_hpo.values())
    try:
        with partial.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                    writer.writerow(FIELDS)
                    cuis = sorted(cui_to_hpo)
                    for start in range(0, len(cuis), batch_size):
                        batch = cuis[start:start + batch_size]
                        query = ("SELECT term,cui,sab,code,aui,tty,is_pref FROM terms WHERE cui IN ("
                                 + ",".join("?" for _ in batch) + ") AND sab IN ("
                                 + ",".join("?" for _ in SOURCES) + ") AND lat='ENG' AND suppress='N' ORDER BY cui,sab,aui")
                        for term, cui, sab, code, aui, tty, preferred in db.execute(query, [*batch, *SOURCES]):
                            if not term.strip():
                                continue
                            for hpo_id in sorted(cui_to_hpo[cui]):
                                if rows >= max_rows:
                                    raise ValueError("Crosswalk exceeded max_rows; no truncated export is published")
                                writer.writerow((term, hpo_id, cui, sab, code, aui, tty, preferred))
                                rows += 1
                                source_counts[sab] += 1
        after = database.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Source database changed during export")
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(), "schema_version": 1,
            "source_database": str(database), "source_database_bytes": before.st_size,
            "source_database_sha256_recomputed": False, "source_mode": "read-only SQLite transaction",
            "hpo_path": str(ontology_path), "hpo_sha256": sha256(ontology_path), "hpo_version": hpo_version,
            "allowed_ids_sha256": hashlib.sha256("\n".join(sorted(allowed_ids)).encode()).hexdigest(),
            "allowed_hpo_ids": len(allowed_ids), "covered_hpo_ids": len(covered),
            "missing_hpo_ids": sorted(allowed_ids - covered), "root_excluded": "HP:0000118",
            "umls_hpo_source_versions": source_versions, "anchor_cuis": len(cui_to_hpo),
            "maximum_hpo_ids_per_cui": max(map(len, cui_to_hpo.values())),
            "filters": {"lat": "ENG", "suppress": "N", "sources": list(SOURCES)},
            "exported_rows": rows, "rows_per_source": dict(sorted(source_counts.items())),
            "output_sha256": sha256(partial), "output_bytes": partial.stat().st_size,
            "export_script_sha256": sha256(Path(__file__)), "batch_size": batch_size,
            "semantics": "Every row is a same-CUI candidate, not an approved equivalence or a compound phenotype. No training or target annotations used.",
        }
        partial.replace(output)
        manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        db.close()
        if partial.exists():
            partial.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=1000000)
    args = parser.parse_args()
    result = export_terms(args.database, args.ontology, args.output, batch_size=args.batch_size, max_rows=args.max_rows)
    print(json.dumps({key: result[key] for key in ("exported_rows", "output_bytes", "output_sha256", "anchor_cuis", "covered_hpo_ids")}, indent=2))


if __name__ == "__main__":
    main()
