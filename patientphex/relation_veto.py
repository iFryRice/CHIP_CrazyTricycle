"""Conservative relation filtering with exact baseline entity preservation."""

import copy


def filter_existing(baseline, candidate):
    if any(baseline.get(key) != candidate.get(key) for key in ["pmc_id", "pmid", "entities"]):
        raise ValueError("Relation filtering requires identical documents and entities.")
    def associations(record):
        result = {item["patient_id"]: set(item["phenotype"]) for item in record["association"]}
        if len(result) != len(record["association"]):
            raise ValueError("Duplicate patient association record.")
        return result
    old, allowed = associations(baseline), associations(candidate)
    if set(old) != set(allowed):
        raise ValueError("Patient coverage differs between baseline and candidate.")
    result = copy.deepcopy(baseline)
    for item in result["association"]:
        item["phenotype"] = [value for value in item["phenotype"] if value in allowed[item["patient_id"]]]
    return result
