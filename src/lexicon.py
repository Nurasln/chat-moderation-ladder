"""Classical NLP step 3 - an edit-distance lexicon matcher (rule baseline).

1. Build a lexicon of offensive words FROM THE TRAINING DATA: a word goes in if it
   appears often enough and mostly in offensive messages.
2. A message is "offensive" if any of its words is close (edit distance) to a
   lexicon word. This catches misspellings like "idiiot" or "fcking".

The same matcher is used by the app to decide WHICH words to mask.
"""
from collections import Counter

from src.edit_distance import edit_distance


def build_lexicon(texts, labels, min_count=8, min_ratio=0.75):
    """Return words that appear >= min_count times and are offensive in >= min_ratio of cases."""
    total, offensive = Counter(), Counter()
    for text, label in zip(texts, labels):
        for word in set(text.split()):        # set(): count each word once per message
            total[word] += 1
            offensive[word] += label
    return sorted(
        w for w, n in total.items()
        if n >= min_count and offensive[w] / n >= min_ratio and w.isalpha() and len(w) >= 3
    )


class LexiconMatcher:
    def __init__(self, lexicon):
        self.lexicon = list(lexicon)
        self.exact = set(lexicon)

    def _allowed_typos(self, word):
        # Short words get no typo tolerance, otherwise "hat" would match "hate"
        return 0 if len(word) <= 4 else 1

    def match(self, word):
        """Return the lexicon word this word matches, or None."""
        if word in self.exact:
            return word
        k = self._allowed_typos(word)
        if k == 0:
            return None
        for lex_word in self.lexicon:
            if len(lex_word) > 4 and edit_distance(word, lex_word, max_dist=k) <= k:
                return lex_word
        return None

    def offensive_words(self, normalized_text):
        """List of (word, matched_lexicon_word) pairs found in a normalized message."""
        hits = []
        for word in normalized_text.split():
            m = self.match(word)
            if m:
                hits.append((word, m))
        return hits

    def predict(self, normalized_texts):
        return [1 if self.offensive_words(t) else 0 for t in normalized_texts]
