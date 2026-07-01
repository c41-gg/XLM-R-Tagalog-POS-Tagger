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

    def parse(self, output: str) -> list[tuple[str, str]]:

        parsed = []

        for line in output.splitlines():

            line = line.strip()

            if not line:
                continue

            # Each line may contain multiple word|tag pairs
            for pair in line.split():

                if "|" not in pair:
                    print(f"Skipping malformed output: {pair}")
                    continue

                word, tag = pair.rsplit("|", 1)

                parsed.append((word, tag))

        return parsed

    def tag(self, tokens: list[TaggedToken]) -> list[TaggedToken]:

        if not tokens:
            return tokens

        sentence = " ".join(t.token for t in tokens)

        with tempfile.NamedTemporaryFile(
            mode="w+",
            delete=False,
            suffix=".txt",
            encoding="utf-8"
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

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8"
            )

            if process.returncode != 0:
                raise RuntimeError(process.stderr)

            parsed = self.parse(process.stdout)

            # ---------- Alignment Check ----------
            if len(parsed) != len(tokens):

                print("\n========== ALIGNMENT ERROR ==========")
                print("Sentence:")
                print(sentence)
                print()

                print(f"Input Tokens : {len(tokens)}")
                print(f"Parsed Tokens: {len(parsed)}")

                print("\nJava Output:")
                print(process.stdout)

                print("=====================================\n")

            # ---------- Assign Tags ----------

            for i, token in enumerate(tokens):

                if i < len(parsed):

                    token.mgnn_tag = parsed[i][1]

                else:
                    token.mgnn_tag = None

            return tokens

        finally:

            os.unlink(temp_file_path)