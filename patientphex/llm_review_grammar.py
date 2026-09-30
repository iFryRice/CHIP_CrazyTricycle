"""A finite JSON grammar for relation decisions and actual source excerpt IDs."""

from itertools import combinations
import json


class ReviewGrammar:
    def __init__(self, tokenizer, excerpt_count):
        if excerpt_count < 1:
            raise ValueError("Source evidence must contain at least one excerpt.")
        references = [(index,) for index in range(excerpt_count)] + list(combinations(range(excerpt_count), 2))
        self.responses = [json.dumps({"decision": decision, "evidence_ids": list(ids)}, separators=(",", ":"))
                          for decision in ["supported", "other_or_general", "unclear"]
                          for ids in ([()] + references if decision == "unclear" else references)]
        self.sequences = []
        self.trie = {}
        for response in self.responses:
            sequence = tokenizer(response, add_special_tokens=False)["input_ids"]
            if tokenizer.decode(sequence, skip_special_tokens=False) != response:
                raise ValueError("The grammar tokenizer does not round-trip the exact JSON.")
            sequence = list(sequence) + [tokenizer.eos_token_id]
            self.sequences.append(sequence)
            node = self.trie
            for token in sequence:
                node = node.setdefault(token, {})
        self.maximum_tokens = max(map(len, self.sequences))

    def next_tokens(self, generated):
        node = self.trie
        for token in generated:
            if int(token) not in node:
                raise ValueError("Generation left the declared finite JSON grammar.")
            node = node[int(token)]
        if not node:
            raise ValueError("Generation continued after the grammar's terminal EOS.")
        return sorted(node)

    def callback(self, prompt_tokens):
        def allowed(batch_id, input_ids):
            if batch_id != 0:
                raise ValueError("This grammar is declared for batch size one.")
            return self.next_tokens(input_ids[prompt_tokens:].tolist())
        return allowed
