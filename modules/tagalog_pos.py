import subprocess
import tempfile
import os

from modules.token_types import TaggedToken, JavaTaggedToken
from modules.alignment import align


class FSPOSTTagger:

    def __init__(self, jar_path: str, model_path: str):

        self.jar_path = jar_path
        self.model_path = model_path

        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"JAR file not found: {jar_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

    # ---------------------------------------------------------
    # Parse Java output
    # ---------------------------------------------------------

    def parse(self, output: str) -> list[JavaTaggedToken]:

        parsed = []

        for line in output.splitlines():

            line = line.strip()

            if not line:
                continue

            for pair in line.split():

                if "|" not in pair:
                    print("Skipping malformed output:", repr(pair))
                    continue

                try:

                    word, tag = pair.rsplit("|", 1)

                    parsed.append(
                        JavaTaggedToken(
                            token=word,
                            tag=tag
                        )
                    )

                except ValueError:

                    print("Malformed pair:", repr(pair))

        return parsed

    # ---------------------------------------------------------
    # Tag sentence
    # ---------------------------------------------------------

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

            # -------------------------------------------------
            # DEBUG
            # -------------------------------------------------

            # Uncomment this temporarily
            # print(repr(process.stdout))

            parsed = self.parse(process.stdout)

            success = align(tokens, parsed)

            if not success:

                print("\n========== ALIGNMENT ERROR ==========")
                print("Sentence:")
                print(sentence)

                print()

                print("Input Tokens :", len(tokens))
                print("Parsed Tokens:", len(parsed))

                print("\nRaw Java Output:")
                print(repr(process.stdout))

                print("\nReadable Java Output:")
                print(process.stdout)

                print("=====================================\n")

            return tokens

        finally:

            os.unlink(temp_file_path)