from __future__ import annotations

from dataclasses import replace

from m2t.models import Segment

_CHINESE = {"zh", "chinese", "cantonese", "mandarin", "yue"}


def is_chinese(language: str | None) -> bool:
    return bool(language) and language.lower() in _CHINESE


def to_traditional(segments: list[Segment]) -> list[Segment]:
    from opencc import OpenCC

    cc = OpenCC("s2twp")
    return [
        replace(s, text=cc.convert(s.text), words=[replace(w, text=cc.convert(w.text)) for w in s.words])
        for s in segments
    ]
