"""Read the supplied HPO release without downloading external resources."""

from collections import defaultdict, deque
from pathlib import Path
import re


class Ontology:
    """A minimal OBO reader restricted to phenotypic abnormalities."""

    def __init__(self, path: str | Path):
        self.terms: dict[str, dict] = {}
        self.version = "unknown"
        self.aliases: dict[str, str] = {}
        current = None

        def finish():
            if current and current.get("id"):
                self.terms[current["id"]] = current

        with Path(path).open(encoding="utf-8") as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if line.startswith("data-version: "):
                    self.version = line.split("/", maxsplit=2)[-1]
                if line == "[Term]":
                    finish()
                    current = {
                        "synonyms": [], "exact_synonyms": [], "parents": [],
                        "alt_ids": [], "obsolete": False,
                    }
                elif line.startswith("["):
                    finish()
                    current = None
                elif current is not None:
                    if line.startswith("id: "):
                        current["id"] = line[4:]
                    elif line.startswith("name: "):
                        current["name"] = line[6:]
                    elif line.startswith("is_a: "):
                        current["parents"].append(line[6:].split()[0])
                    elif line.startswith("alt_id: "):
                        current["alt_ids"].append(line[8:])
                    elif line == "is_obsolete: true":
                        current["obsolete"] = True
                    elif line.startswith("synonym: "):
                        match = re.match(r'synonym: "((?:\\.|[^"\\])*)"\s+(\w+)', line)
                        if match:
                            synonym = re.sub(r"\\(.)", r"\1", match[1])
                            current["synonyms"].append(synonym)
                            if match[2] == "EXACT":
                                current["exact_synonyms"].append(synonym)
            finish()

        children: dict[str, set[str]] = defaultdict(set)
        for identifier, term in self.terms.items():
            if not term["obsolete"]:
                for parent in term["parents"]:
                    children[parent].add(identifier)
                for alias in term["alt_ids"]:
                    self.aliases[alias] = identifier
        self.allowed_ids: set[str] = set()
        queue = deque(children["HP:0000118"])
        while queue:
            identifier = queue.popleft()
            if identifier not in self.allowed_ids:
                self.allowed_ids.add(identifier)
                queue.extend(children[identifier])

    def normalize_identifier(self, identifier: str) -> str | None:
        """Preserve compound annotation IDs while rejecting out-of-branch IDs."""
        result = []
        for part in identifier.split(";"):
            part = part.strip()
            canonical = self.aliases.get(part, part)
            if canonical != "-1" and canonical not in self.allowed_ids:
                return None
            if canonical not in result:
                result.append(canonical)
        return ";".join(result) if result else None
