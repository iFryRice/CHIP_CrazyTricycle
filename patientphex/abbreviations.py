"""Resolve abbreviations from definitions in the same original document.

Parenthetical candidate alignment follows the backwards character matching
idea of Schwartz and Hearst (2003), with stricter short-form filtering.
"""

from collections import defaultdict
import re

from .entities import is_negated
from .span_linking import _strict_key


def short_form(text):
    letters = [character for character in text if character.isalpha()]
    return (2 <= len(text) <= 12 and bool(letters) and text[0].isalnum()
            and sum(character.isupper() for character in letters) >= 2
            and re.fullmatch(r"[A-Za-z0-9/+\-]+", text) is not None)


def align_definition(short, long):
    """Return the start of a short form's minimal suffix definition, or None."""
    characters = [character.lower() for character in short if character.isalnum()]
    position = len(long) - 1
    for index in range(len(characters) - 1, -1, -1):
        while position >= 0 and (long[position].lower() != characters[index]
                                or index == 0 and position > 0 and long[position-1].isalnum()):
            position -= 1
        if position < 0:
            return None
        position -= 1
    start = position + 1
    selected = long[start:]
    if len(selected) <= len(short) or len(selected.split()) > min(len(characters) + 5, len(characters) * 2):
        return None
    return start


def extract_definitions(document):
    """Use only text, preserving the exact global offsets for both forms."""
    definitions = []
    seen = set()
    for paragraph in document["full_text"]:
        text = paragraph["text"]
        for match in re.finditer(r"\(([^()\n]{2,180})\)", text):
            inner = match.group(1).strip()
            prefix_end = match.start()
            prefix = text[:prefix_end].rstrip()
            prefix_start = max([0] + [m.end() for m in re.finditer(r"[.!?;\n]", prefix)])
            before = prefix[prefix_start:]
            if short_form(inner):
                short = inner
                short_start = match.start(1) + len(match.group(1)) - len(match.group(1).lstrip())
                long = before
                long_base = prefix_start
            else:
                previous = re.search(r"([A-Za-z0-9/+\-]+)$", prefix)
                if previous is None or not short_form(previous.group(1)):
                    continue
                short, short_start = previous.group(1), previous.start(1)
                long = inner
                long_base = match.start(1) + len(match.group(1)) - len(match.group(1).lstrip())
            start = align_definition(short, long)
            if start is None:
                continue
            long_text = long[start:].rstrip()
            long_start = long_base + start
            identity = (paragraph["offset"] + short_start, paragraph["offset"] + long_start, short, long_text)
            if identity in seen:
                continue
            seen.add(identity)
            definitions.append({"short": short, "long": long_text, "short_offset": identity[0], "long_offset": identity[1],
                                "long_length": len(long_text)})
    return definitions


def resolve_definitions(document, linker):
    grouped = defaultdict(list)
    for definition in extract_definitions(document):
        grouped[definition["short"]].append({**definition, "resolution": linker.resolve(definition["long"])})
    resolved = {}
    for short, definitions in sorted(grouped.items()):
        identifiers = {entry["resolution"]["identifier"] for entry in definitions}
        ambiguous = any(entry["resolution"].get("ambiguous") for entry in definitions)
        if not ambiguous and len(identifiers) == 1 and "-1" not in identifiers:
            status, identifier = "mapped", next(iter(identifiers))
        elif (not ambiguous and len({_strict_key(entry["long"]) for entry in definitions}) == 1
              and all(entry["resolution"]["source"] == "no_candidate" for entry in definitions)):
            status, identifier = "unresolved_definition", None
        else:
            status, identifier = "abstain", None
        resolved[short] = {"status": status, "identifier": identifier, "definitions": definitions}
    return resolved


def apply_definitions(document, entities, resolved, *, add_missing=False, suppress_unresolved=False):
    """Correct local senses; optionally add mapped or remove unsupported acronyms."""
    if document.get("entities") or document.get("association"):
        raise ValueError("Abbreviation prediction requires a blind document.")
    output, changes = [], []
    for entity in entities:
        entry = resolved.get(entity["text"])
        if entry and entry["status"] == "unresolved_definition" and suppress_unresolved:
            changes.append({"action": "remove", "original": entity})
        elif entry and entry["status"] == "mapped" and entity["identifier"] != entry["identifier"]:
            revised = {**entity, "identifier": entry["identifier"]}
            output.append(revised)
            changes.append({"action": "correct", "original": entity, "revised": revised})
        else:
            output.append(entity)
    if add_missing:
        for short, entry in sorted(resolved.items()):
            if entry["status"] != "mapped":
                continue
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(short) + r"(?![A-Za-z0-9])")
            for paragraph in document["full_text"]:
                for match in pattern.finditer(paragraph["text"]):
                    start, end = paragraph["offset"] + match.start(), paragraph["offset"] + match.end()
                    if any(start < entity["offset"] + entity["length"] and entity["offset"] < end for entity in output):
                        continue
                    entity = {"identifier": entry["identifier"], "type": "Phenotype", "offset": start, "length": end-start,
                              "text": match.group(), "note": "NO" if is_negated(paragraph["text"], match.start(), match.end()) else None}
                    output.append(entity)
                    changes.append({"action": "add", "revised": entity})
    return sorted(output, key=lambda entity: (entity["offset"], entity["length"])), changes
