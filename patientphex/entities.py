"""Deterministic CPU phenotype recognition with a learned candidate filter."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
import unicodedata

from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression

from .ontology import Ontology


WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
GAP = re.compile(r"^[\s,()/\-‐‑‒–—−]*$")
NEGATION = re.compile(
    r"\b(?:no|without|denied|denies|denying|absence of|absent of|"
    r"negative for|free of|not (?:have|having|show|showing|exhibit|exhibiting|"
    r"reveal|revealing|associated with|accompanied by|present with)|"
    r"neither)\b", re.I,
)
CONTRAST = re.compile(r"\b(?:but|however|except|although|whereas|yet)\b", re.I)
SUFFIX_NEGATION = re.compile(
    r"^\s*(?:\([^)]{0,25}\)\s*)?(?:(?:was|were|is|are|has been|had been)\s+)?"
    r"(?:absent|not (?:present|observed|detected|seen|found|noted)|ruled out)\b", re.I,
)


def normalize_token(token: str) -> str:
    """Normalize matching tokens without changing original character positions."""
    token = unicodedata.normalize("NFKC", token).casefold().replace("’", "'")
    if token.endswith("'s"):
        token = token[:-2]
    irregular = {"feet": "foot", "teeth": "tooth", "abnormalities": "abnormality",
                 "deformities": "deformity", "anomalies": "anomaly"}
    if token in irregular:
        return irregular[token]
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(
        ("ss", "is", "us", "ous", "ics", "itis", "sis", "etes")
    ):
        return token[:-1]
    return token


def phrase_key(text: str) -> tuple[str, ...]:
    return tuple(normalize_token(match[0]) for match in WORD.finditer(text))


def is_negated(text: str, start: int, end: int) -> bool:
    """Recognize local explicit denial, including simple coordinated lists."""
    prefix = text[max(0, start - 180):start]
    prefix = re.split(r"[.!?;:\n]", prefix)[-1]
    prefix = CONTRAST.split(prefix)[-1]
    if not re.search(r"\bno(?:t)?\s+only\b", prefix, re.I):
        matches = list(NEGATION.finditer(prefix))
        if matches:
            tail = prefix[matches[-1].end():]
            if len(tail) <= 125 and not re.search(
                r"\b(?:developed|presented|suffered|diagnosed|revealed|showed|"
                r"exhibited|demonstrated|subsequently|later)\b", tail, re.I
            ):
                return True
    return bool(SUFFIX_NEGATION.match(text[end:end + 90]))


@dataclass
class Candidate:
    paragraph: dict
    start: int
    end: int
    key: tuple[str, ...]
    identifier: str
    dictionary: bool
    abbreviation: bool = False
    score: float = 1.0

    @property
    def text(self):
        return self.paragraph["text"][self.start:self.end]

    @property
    def offset(self):
        return self.paragraph["offset"] + self.start


class EntityExtractor:
    """HPO/training lexicon matcher; ``learned`` filters matches by context.

    Training documents alone supply aliases, disambiguation counts, and filter
    labels. All returned spans are slices of the untouched input paragraphs.
    """

    def __init__(self, ontology: Ontology, mode: str = "learned", threshold: float = 0.42):
        if mode not in {"dictionary", "learned"}:
            raise ValueError(f"Unknown entity mode: {mode}")
        self.ontology = ontology
        self.mode = mode
        self.threshold = threshold
        self.trie = {}
        self.dictionary_ids = defaultdict(set)
        self.training_ids = defaultdict(Counter)
        self.training_docs = defaultdict(set)
        self.lowercase_short_aliases = set()
        self.positive_counts = Counter()
        self.candidate_counts = Counter()
        self.document_positive_counts = defaultdict(Counter)
        self.document_candidate_counts = defaultdict(Counter)
        self.hasher = FeatureHasher(n_features=2 ** 18, input_type="dict", alternate_sign=False)
        self.model = None
        self.stats = {}
        self._build_ontology_lexicon()

    def _insert(self, key):
        node = self.trie
        for token in key:
            node = node.setdefault(token, {})
        node[None] = key

    def _build_ontology_lexicon(self):
        for identifier in sorted(self.ontology.allowed_ids):
            term = self.ontology.terms[identifier]
            for name in [term.get("name", ""), *term["exact_synonyms"]]:
                key = phrase_key(name)
                if key and 2 <= len(name) <= 220 and len(key) <= 24:
                    self.dictionary_ids[key].add(identifier)
                    self._insert(key)

    def fit(self, documents: list[dict]):
        # Reset learned state so fitting twice cannot leak an earlier split.
        self.trie = {}
        self.dictionary_ids.clear()
        self.training_ids.clear()
        self.training_docs.clear()
        self.lowercase_short_aliases.clear()
        self.positive_counts.clear()
        self.candidate_counts.clear()
        self.document_positive_counts.clear()
        self.document_candidate_counts.clear()
        self.model = None
        self._build_ontology_lexicon()
        for document in documents:
            doc_id = document["pmc_id"]
            for entity in document["entities"]:
                identifier = self.ontology.normalize_identifier(entity["identifier"])
                key = phrase_key(entity["text"])
                if identifier and key and len(key) <= 45:
                    self.training_ids[key][identifier] += 1
                    self.training_docs[key].add(doc_id)
                    if len(entity["text"]) <= 3 and entity["text"].islower():
                        self.lowercase_short_aliases.add(key)
                    self._insert(key)

        rows, labels, document_ids = [], [], []
        covered, gold_count = 0, 0
        for document in documents:
            doc_id = document["pmc_id"]
            gold = {(e["offset"], e["length"]) for e in document["entities"]}
            gold_count += len(gold)
            candidates = self._candidates(document)
            candidate_spans = {(c.offset, c.end - c.start) for c in candidates}
            covered += len(gold & candidate_spans)
            for candidate in candidates:
                label = int((candidate.offset, candidate.end - candidate.start) in gold)
                self.candidate_counts[candidate.key] += 1
                self.document_candidate_counts[doc_id][candidate.key] += 1
                if label:
                    self.positive_counts[candidate.key] += 1
                    self.document_positive_counts[doc_id][candidate.key] += 1
                rows.append(candidate)
                labels.append(label)
                document_ids.append(doc_id)

        if len(set(labels)) == 2:
            features = self.hasher.transform(
                self._features(row, document_id=doc_id)
                for row, doc_id in zip(rows, document_ids)
            )
            self.model = LogisticRegression(
                C=0.6, solver="liblinear", max_iter=250, random_state=42,
            )
            self.model.fit(features, labels)
        self.stats = {
            "ontology_version": self.ontology.version,
            "ontology_terms": len(self.ontology.allowed_ids),
            "dictionary_phrases": len(self.dictionary_ids),
            "training_phrases": len(self.training_ids),
            "training_candidates": len(rows),
            "training_positive_candidates": sum(labels),
            "training_candidate_coverage": covered / max(1, gold_count),
            "filter_threshold": self.threshold,
        }
        return self

    def _identifier(self, key):
        if self.training_ids.get(key):
            return self.training_ids[key].most_common(1)[0][0]
        identifiers = self.dictionary_ids.get(key, ())
        return min(identifiers) if identifiers else None

    def _paragraph_candidates(self, paragraph):
        text = paragraph["text"]
        tokens = list(WORD.finditer(text))
        normalized = [normalize_token(token[0]) for token in tokens]
        for i, token in enumerate(tokens):
            node = self.trie
            for j in range(i, min(i + 45, len(tokens))):
                if j > i and not GAP.fullmatch(text[tokens[j - 1].end():tokens[j].start()]):
                    break
                node = node.get(normalized[j])
                if node is None:
                    break
                if None not in node:
                    continue
                key = node[None]
                start, end = token.start(), tokens[j].end()
                mention = text[start:end]
                if len(key) == 1 and len(mention) <= 3:
                    # Short forms must retain their original uppercase spelling.
                    if not mention.isupper() and key not in self.lowercase_short_aliases:
                        continue
                    if len(mention) == 1:
                        continue
                identifier = self._identifier(key)
                if identifier:
                    yield Candidate(paragraph, start, end, key, identifier, key in self.dictionary_ids)

    def _candidates(self, document):
        candidates = [candidate for paragraph in document["full_text"]
                      for candidate in self._paragraph_candidates(paragraph)]
        # Resolve parenthetical abbreviations within this document only.
        abbreviations = {}
        for candidate in candidates:
            if len(candidate.key) < 2 or len(candidate.text) < 8:
                continue
            following = candidate.paragraph["text"][candidate.end:candidate.end + 18]
            match = re.match(r"\s*\(([A-Z][A-Z0-9\-]{1,7})\)", following)
            if not match:
                continue
            abbreviation = match[1]
            letters = re.sub(r"[^A-Za-z]", "", abbreviation).lower()
            initial_letters = "".join(token[0] for token in candidate.key
                                      if token not in {"of", "the", "and", "with", "in"})
            if letters != initial_letters and not self._aligned_abbreviation(letters, candidate.text):
                continue
            previous = abbreviations.get(abbreviation)
            if previous is None or len(candidate.text) > previous[1]:
                abbreviations[abbreviation] = (candidate.identifier, len(candidate.text))
        existing = {(id(c.paragraph), c.start, c.end): c for c in candidates}
        if abbreviations:
            pattern = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in abbreviations) + r")\b")
            for paragraph in document["full_text"]:
                for match in pattern.finditer(paragraph["text"]):
                    key = (id(paragraph), match.start(), match.end())
                    identifier = abbreviations[match[0]][0]
                    if key in existing:
                        existing[key].identifier = identifier
                        existing[key].abbreviation = True
                    else:
                        candidate = Candidate(paragraph, match.start(), match.end(), phrase_key(match[0]),
                                              identifier, False, abbreviation=True)
                        candidates.append(candidate)
                        existing[key] = candidate
        return candidates

    @staticmethod
    def _aligned_abbreviation(abbreviation, long_form):
        if not 2 <= len(abbreviation) <= 7:
            return False
        long_form = long_form.lower()
        position = len(long_form)
        for index, letter in reversed(list(enumerate(abbreviation))):
            position = long_form.rfind(letter, 0, position)
            if position < 0:
                return False
            if index == 0 and position > 0 and long_form[position - 1].isalnum():
                return False
        return True

    def _features(self, candidate, document_id=None):
        text = candidate.paragraph["text"]
        mention = candidate.text
        key = candidate.key
        positive = self.positive_counts[key]
        total = self.candidate_counts[key]
        training_documents = len(self.training_docs[key])
        if document_id is not None:
            positive -= self.document_positive_counts[document_id][key]
            total -= self.document_candidate_counts[document_id][key]
            training_documents -= int(document_id in self.training_docs[key])
        rate = (positive + 1.5) / (total + 3.0)
        features = {
            "bias": 1.0,
            "length": min(len(mention), 100) / 30,
            "tokens": min(len(key), 15) / 5,
            "dictionary": float(candidate.dictionary),
            "abbreviation": float(candidate.abbreviation),
            "uppercase": float(mention.isupper()),
            "short": float(len(mention) <= 4),
            "short_uppercase": float(len(mention) <= 4 and mention.isupper()),
            "has_digit": float(any(c.isdigit() for c in mention)),
            "observed_rate": rate,
            "observed_count": math.log1p(total),
            "observed_positive": math.log1p(positive),
            "observed_documents": math.log1p(training_documents),
            "section:" + str(candidate.paragraph.get("section_type", "")).lower(): 1,
            "paragraph_type:" + str(candidate.paragraph.get("type", "")).lower(): 1,
            "mention:" + " ".join(key): 1,
            "identifier:" + candidate.identifier: 1,
            "negated": float(is_negated(text, candidate.start, candidate.end)),
        }
        if total:
            features["known_low_rate"] = float(rate < 0.25)
            features["known_high_rate"] = float(rate > 0.75)
        for word in key:
            features["mention_word:" + word] = 1
        compact = " " + " ".join(key) + " "
        for width in (3, 4):
            for start in range(len(compact) - width + 1):
                features[f"char{width}:" + compact[start:start + width]] = 0.25
        before = list(WORD.finditer(text[max(0, candidate.start - 120):candidate.start]))[-7:]
        after = list(WORD.finditer(text[candidate.end:candidate.end + 100]))[:6]
        for index, token in enumerate(reversed(before)):
            word = normalize_token(token[0])
            features["before:" + word] = 1 / (1 + index / 3)
            if index < 2:
                features[f"before{index}:" + word] = 1
        for index, token in enumerate(after):
            word = normalize_token(token[0])
            features["after:" + word] = 1 / (1 + index / 3)
            if index < 2:
                features[f"after{index}:" + word] = 1
        return features

    def predict(self, document: dict) -> list[dict]:
        candidates = self._candidates(document)
        if self.mode == "learned" and self.model is not None and candidates:
            matrix = self.hasher.transform(self._features(candidate) for candidate in candidates)
            scores = self.model.predict_proba(matrix)[:, 1]
            for candidate, score in zip(candidates, scores):
                candidate.score = float(score)
            candidates = [c for c in candidates if c.score >= self.threshold]
        elif self.mode == "dictionary":
            candidates = [c for c in candidates if not (
                len(c.text) <= 3 and not c.abbreviation and
                self.positive_counts[c.key] == 0
            )]

        # Prefer the longest clinical phrase when dictionary matches nest.
        accepted = []
        for candidate in sorted(candidates, key=lambda c: (-(c.end - c.start), -c.score, c.offset)):
            if any(candidate.offset < other.offset + other.end - other.start and
                   other.offset < candidate.offset + candidate.end - candidate.start
                   for other in accepted):
                continue
            accepted.append(candidate)
        return [
            {"identifier": candidate.identifier, "type": "Phenotype", "offset": candidate.offset,
             "length": candidate.end - candidate.start, "text": candidate.text,
             "note": "NO" if is_negated(candidate.paragraph["text"], candidate.start, candidate.end) else None}
            for candidate in sorted(accepted, key=lambda c: (c.offset, c.end - c.start))
        ]
