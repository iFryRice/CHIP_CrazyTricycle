"""Expose provided patient identities beside source text without altering it."""

import copy


def mark_task(task, patients):
    result = copy.deepcopy(task)
    references = []
    spans = []
    for patient in patients:
        role = "TARGET" if patient["patient_id"] == task["patient_id"] else "OTHER"
        references.append({"patient_id": patient["patient_id"], "role_for_this_question": role,
                           "mentions": [{"offset": m["offset"], "length": m["length"], "text": m["text"]} for m in patient["mention"]]})
        spans.extend((m, f"{role}:{patient['patient_id']}") for m in patient["mention"])
    if sum(reference["role_for_this_question"] == "TARGET" for reference in references) != 1:
        raise ValueError("Exactly one provided patient must be the target.")
    result["patient_references"] = references
    for fragment in result["fragments"]:
        source = fragment["text"]
        if "[TARGET:" in source or "[OTHER:" in source:
            raise ValueError("Synthetic patient markers would collide with source text.")
        events, labels = {}, []
        for span, label in spans:
            start = span["offset"]-fragment["offset"]
            end = start+span["length"]
            if start < 0 or end > len(source):
                continue
            if source[start:end] != span["text"]:
                raise ValueError("A provided patient marker does not match the original source.")
            events.setdefault(start, []).append((1, f"[{label}]"))
            events.setdefault(end, []).append((0, f"[/{label}]"))
            labels.append({"start_in_original_fragment": start, "end_in_original_fragment": end, "label": label})
        pieces = []
        for offset in range(len(source)+1):
            pieces.extend(value for _, value in sorted(events.get(offset, [])))
            if offset < len(source):
                pieces.append(source[offset])
        fragment["patient_annotated_text"] = "".join(pieces)
        fragment["patient_annotations"] = labels
        fragment["annotation_legend"] = "Added TARGET/OTHER tags label provided patient references only. Original text and offsets are unchanged in the text field. Patient IDs are arbitrary and may denote relatives, not the primary case."
        fragment["target_patient_id"] = task["patient_id"]
    result["patient_marker_protocol"] = "provided_anchor_labels"
    return result
