import re
from modules.token_types import TaggedToken

TOKEN_PATTERN = re.compile(
    r"\.\.\.|"
    r"n't|'m|'re|'ve|'ll|'d|'s|"
    r"\w+(?:[-']\w+)*|"
    r"[^\w\s]",
    re.UNICODE
)

# Matches 2+ periods with optional whitespace between them, e.g.
# "hehe . . . nako" -- Stanford's tagger collapses any such run into a
# single ellipsis token, but our tokenizer only recognizes literal "..."
# (no spaces). Left unhandled, "hehe . . ." produces three separate "."
# TaggedTokens while the Java side produces one "...", permanently
# desyncing the aligner for the rest of the sentence.
SPACED_ELLIPSIS_PATTERN = re.compile(r"(?:\.\s*){2,}")

# English contraction suffixes that Stanford's PTBTokenizer (used by
# FSPOST) always splits off as their own token, e.g. "don't" -> "do" +
# "n't", "I'm" -> "I" + "'m". Our tokenizer previously kept these as one
# token, which is fine for Tagalog-side apostrophe forms (n'ya, s'ya,
# t'yak -- structurally distinct, single-consonant + vowel-suffix) but
# permanently desyncs alignment on any English/Taglish contraction,
# which shows up constantly given the code-switching in this corpus.
CONTRACTION_PATTERN = re.compile(
    r"(?i)(\w+?)(n't|'m|'re|'ve|'ll|'d|'s)\b"
)


def _split_contractions(sentence: str) -> str:
    return CONTRACTION_PATTERN.sub(r"\1 \2", sentence)


def _normalize_ellipsis(sentence: str) -> str:
    return SPACED_ELLIPSIS_PATTERN.sub("...", sentence)


def _strip_soft_hyphens(sentence: str) -> str: # U+00AD SOFT HYPHEN
    return sentence.replace("\u00ad", "")


def tokenize(sentence: str) -> list[TaggedToken]:

    sentence = _strip_soft_hyphens(sentence)
    sentence = _normalize_ellipsis(sentence)
    sentence = _split_contractions(sentence)

    return [
        TaggedToken(index=i, token=t)
        for i, t in enumerate(TOKEN_PATTERN.findall(sentence))
    ]

def detokenize(tokens: list[TaggedToken]) -> str:

    words = [t.token for t in tokens]

    sentence = ""

    for word in words:

        if not sentence:
            sentence = word

        elif re.fullmatch(r"[^\w\s]", word):
            sentence += word

        else:
            sentence += " " + word

    return sentence