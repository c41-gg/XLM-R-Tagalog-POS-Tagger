import re
from modules.token_types import TaggedToken

TOKEN_PATTERN = re.compile(
    r"\w+(?:[-']\w+)*|[^\w\s]",
    re.UNICODE
)



def tokenize(sentence: str) -> list[TaggedToken]:

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
   