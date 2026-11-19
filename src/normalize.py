"""Classical NLP step 1 - text normalization with regular expressions.

Chat / tweet text is noisy. The same insult can be written many ways:
    "BULLSH1T", "bullshiiiiit", "b.u.l.l.s.h.i.t" ...
Normalizing first means every model sees a cleaner, more uniform input.

All models use THIS function, so they are compared on the same input.
"""
import html
import re

# Leetspeak: symbols/digits that people use instead of letters
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})

# Tokens with digits that are NOT leetspeak: 1st, 2nd, 3rd, 9pm, 4k, 80s ...
NUMBER_WORD = re.compile(r"^\d+(st|nd|rd|th|am|pm|s|k|m)?$")


def _unleet(token: str) -> str:
    """Translate leetspeak in one token, only if it looks like a word ('sh1t' yes, '2nd' no)."""
    token = token.strip("!?.,;:\"'()")              # "liar!!!" -> "liar" (end punctuation is not leet)
    has_letter = any(c.isalpha() for c in token)
    has_leet = any(c in "013457@$!" for c in token)
    if not (has_letter and has_leet) or NUMBER_WORD.match(token):
        return token
    return token.translate(LEET)


def normalize(text: str) -> str:
    text = html.unescape(text)                      # "&amp;" -> "&"
    text = text.lower()
    text = re.sub(r"@user\b", " ", text)            # anonymized mentions carry no meaning
    text = re.sub(r"\burl\b|https?://\S+", " ", text)
    text = text.replace("#", " ")                   # "#maga" -> "maga" (keep the word)
    text = re.sub(r"(?<=[a-z])\*+(?=[a-z])", "", text)            # "f*cking" -> "fcking"
    text = re.sub(r"\b(?:[a-z][.\-_]){2,}[a-z]\b",                # "b.u.l.l.s.h.i.t" -> "bullshit"
                  lambda m: re.sub(r"[.\-_]", "", m.group()), text)

    tokens = [_unleet(t) for t in text.split()]
    text = " ".join(tokens)

    text = re.sub(r"(\w)\1{2,}", r"\1\1", text)     # "sooooo" -> "soo" (max 2 repeats)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)       # drop remaining punctuation / emoji
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    # Quick self-check:  python -m src.normalize
    examples = [
        "@user You are a BULLSH1T liar!!! #MAGA URL",
        "shuuuuut up you f*cking id10t",
        "this is b.u.l.l.s.h.i.t &amp; you know it",
        "See you on the 2nd at 9pm",
    ]
    for e in examples:
        print(f"{e!r:55} -> {normalize(e)!r}")
