from __future__ import annotations

from dataclasses import dataclass

from m2t.models import Word


@dataclass
class ASRResult:
    words: list[Word]
    language: str
