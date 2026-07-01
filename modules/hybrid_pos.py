from modules.tokenizer import tokenize

from modules.english_pos import EnglishPOSTagger

from modules.tagalog_pos import FSPOSTTagger

from modules.tag_mapping import english_to_mgnn

from dataclasses import dataclass

class HybridPOSTagger:

    def __init__(self,java_jar,tagalog_model):

        self.tl = FSPOSTTagger(
            java_jar,
            tagalog_model
        )

        self.en = EnglishPOSTagger()

    def tag(self, sentence):

        tokens = tokenize(sentence)

        self.tl.tag(tokens)

        fw_indices = [
            i for i, t in enumerate(tokens)
            if t.mgnn_tag == "FW"
        ]

        if not fw_indices:
            return tokens
        
        fw_words = [tokens[i] for i in fw_indices]

        english_tags = self.en.tag(fw_words)

        for idx, en_token in zip(fw_indices, english_tags):

            tokens[idx].mgnn_tag = english_to_mgnn(
                en_token.token,
                en_token.penn_tag
            )

        return tokens