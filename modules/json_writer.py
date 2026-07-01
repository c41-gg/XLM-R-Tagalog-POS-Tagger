import json
from pathlib import Path
from modules.token_types import TaggedToken


class JSONDatasetWriter:

    def __init__(self, output_path: str):

        self.output_path = Path(output_path)
        self.data = []

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def add_sentence(self, tokens: list[TaggedToken]):

        self.data.append({
            "tokens": [t.token for t in tokens],
            "labels": [t.mgnn_tag for t in tokens]
        })

    def save(self):

        self.output_path.write_text(
            json.dumps(
                self.data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )