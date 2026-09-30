"""Constrained decoding cannot invent IDs, exceed the schema or omit evidence."""

import unittest

from patientphex.llm_association_references import parse_decision
from patientphex.llm_review_grammar import ReviewGrammar


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class ReviewGrammarTests(unittest.TestCase):
    def test_every_legal_completion_is_valid_and_its_prefixes_are_allowed(self):
        grammar = ReviewGrammar(CharacterTokenizer(), 4)
        self.assertEqual(len(grammar.responses), 31)
        fragments = [{"text": f"source {i}"} for i in range(4)]
        for text, sequence in zip(grammar.responses, grammar.sequences, strict=True):
            parsed = parse_decision(text, fragments)
            self.assertIn(parsed["decision"], {"supported", "other_or_general", "unclear"})
            for index, token in enumerate(sequence):
                self.assertIn(token, grammar.next_tokens(sequence[:index]))
        with self.assertRaisesRegex(ValueError, "left the declared"):
            grammar.next_tokens([999999])

    def test_single_excerpt_still_allows_uncertainty_without_forced_support(self):
        grammar = ReviewGrammar(CharacterTokenizer(), 1)
        self.assertEqual(len(grammar.responses), 4)
        self.assertIn('{"decision":"unclear","evidence_ids":[]}', grammar.responses)
        self.assertIn('{"decision":"other_or_general","evidence_ids":[0]}', grammar.responses)
        with self.assertRaises(ValueError):
            ReviewGrammar(CharacterTokenizer(), 0)


if __name__ == "__main__":
    unittest.main()
