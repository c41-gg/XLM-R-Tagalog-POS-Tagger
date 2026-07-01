from dataclasses import dataclass


@dataclass
class TaggedToken:
    index: int
    
    token: str

    mgnn_tag: str | None = None

    penn_tag: str | None = None