import subprocess
import tempfile
import os

from modules.token_types import TaggedToken


class FSPOSTTagger:

    def __init__(self, jar_path: str, model_path: str):

        self.jar_path = jar_path
        self.model_path = model_path

        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"JAR file not found: {jar_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def tag(self, tokens: list[TaggedToken]) -> list[TaggedToken]:

        if not tokens:
            return tokens

        sentence = " ".join(t.token for t in tokens)

        with tempfile.NamedTemporaryFile(
            mode="w+",
            delete=False,
            suffix=".txt"
        ) as temp_file:

            temp_file.write(sentence)
            temp_file_path = temp_file.name

        try:
            command = [
                "java",
                "-mx1g",
                "-cp",
                self.jar_path,
                "edu.stanford.nlp.tagger.maxent.MaxentTagger",
                "-model",
                self.model_path,
                "-textFile",
                temp_file_path
            ]

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            output, error = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Tagger failed: {error}")

            tagged_output = output.strip().split()

            parsed = []

            for item in tagged_output:

                if "|" not in item:
                    continue

                word, tag = item.rsplit("|", 1)

                parsed.append((word, tag))

            # ALIGN back to original tokens
            for token, (_, tag) in zip(tokens, parsed):
                token.mgnn_tag = tag

            return tokens

        finally:
            os.unlink(temp_file_path)