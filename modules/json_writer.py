import json
from pathlib import Path
from modules.token_types import TaggedToken
from modules.mgnn_schema import decompose


class JSONDatasetWriter:

    def __init__(self, output_path: str):

        self.output_path = Path(output_path)
        self.data = []

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def add_sentence(self, tokens: list[TaggedToken]):

        categories = []
        subtypes = []
        focuses = []
        ligatures = []
        unresolved = []

        for t in tokens:

            tag = t.mgnn_tag

            # Tags like EMOJI / IGNORED aren't real MGNN tags -- decompose()
            # still handles them (routed to category="PIPELINE"), so this
            # stays a no-op passthrough, not a special case here.
            d = decompose(tag) if tag else None

            if d is None or d.unknown_parts:
                # Unrecognized atomic piece(s) -- keep the raw tag visible
                # instead of silently writing nulls, so these are easy to
                # grep for and patch into mgnn_schema.ATOMIC_TAGS later.
                categories.append(None)
                subtypes.append(None)
                focuses.append(None)
                ligatures.append(False)
                unresolved.append({
                    "index": t.index,
                    "token": t.token,
                    "tag": tag,
                    "unknown_parts": d.unknown_parts if d else None
                })
                continue

            categories.append(d.category)
            subtypes.append(d.subtype)
            focuses.append(d.focus)
            ligatures.append(d.ligature)

            if d.extra_subtypes:
                # Same-axis stacking (e.g. JJCS_JJD) -- rare, but flagged
                # per-token rather than silently dropped so you can decide
                # policy (keep first vs. most-specific) with real examples
                # in hand instead of guessing.
                unresolved.append({
                    "index": t.index,
                    "token": t.token,
                    "tag": tag,
                    "extra_subtypes": d.extra_subtypes
                })

        entry = {
            "tokens": [t.token for t in tokens],
            "labels": [t.mgnn_tag for t in tokens],
            "categories": categories,
            "subtypes": subtypes,
            "focuses": focuses,
            "ligatures": ligatures,
        }

        if unresolved:
            entry["unresolved"] = unresolved

        self.data.append(entry)

    def save(self):

        self.output_path.write_text(
            json.dumps(
                self.data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )