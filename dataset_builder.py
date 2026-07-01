from modules.json_writer import JSONDatasetWriter


class DatasetBuilder:

    def __init__(self, tagger, output_file):

        self.tagger = tagger
        self.writer = JSONDatasetWriter(output_file)

    def process_corpus(self, sentences: list[str]):

        for i, sent in enumerate(sentences):

            try:
                tokens = self.tagger.tag(sent)

                self.writer.add_sentence(tokens)

            except Exception as e:

                print(f"[ERROR] Sentence {i}: {sent}")
                print(e)

        self.writer.save()