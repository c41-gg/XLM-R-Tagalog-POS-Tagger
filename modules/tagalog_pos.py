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
    # validate tokens
    # ---------------------------------------------------------

    def validate_tokens(self, tokens):

        for token in tokens:

            if token.token == "":
                print("Empty token!")

            if token.token.isspace():
                print("Whitespace token!")
    
    # ---------------------------------------------------------
    # Tag sentence
    # ---------------------------------------------------------



    def tag(self, tokens: list[TaggedToken]) -> list[TaggedToken]:

        if not tokens:
            return tokens
        
        self.validate_tokens(tokens)

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
                "-mx3g",
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

                # align() mutates `tokens` in place as it goes, so the
                # first token still missing an mgnn_tag is exactly where
                # it gave up -- use that instead of guessing blind from
                # count mismatches alone (several failures have equal
                # input/parsed counts, meaning the desync is positional,
                # not a length problem, and counts alone can't diagnose
                # those).
                failure_index = next(
                    (t.index for t in tokens if t.mgnn_tag is None),
                    None
                )
                failure_token = (
                    tokens[failure_index].token
                    if failure_index is not None else "?"
                )

                java_dump = " ".join(
                    f"{jt.token}|{jt.tag}" for jt in parsed
                )

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

                # Previously this fell through to `return tokens` even on
                # failure, silently shipping a partially-tagged sentence
                # (everything after the desync point keeps mgnn_tag=None).
                # Raise instead so run_phase1.py's existing try/except in
                # process_sentence() routes the whole sentence to the
                # error log, rather than writing null-tagged tokens into
                # the dataset.
                raise RuntimeError(
                    f"Alignment failed at original index {failure_index} "
                    f"(token={failure_token!r}): {len(tokens)} input tokens "
                    f"vs {len(parsed)} parsed tokens. "
                    f"Sentence: {sentence!r} | "
                    f"Full Java parse: {java_dump}"
                )

            return tokens

        finally:

            os.unlink(temp_file_path)