from nltk.tag import PerceptronTagger
from modules.token_types import TaggedToken

class EnglishPOSTagger:

    def __init__(self):

        self.tagger = PerceptronTagger()

    def tag(self, tokens: list[TaggedToken]):

        """
        Returns:

        [
            ("I","PRP"),
            ("love","VBP")
        ]
        """
        tagged = self.tagger.tag(
            [t.token for t in tokens]
        )

        for token, (_, penn) in zip(tokens, tagged):

            token.penn_tag = penn
        
        return tokens
        
