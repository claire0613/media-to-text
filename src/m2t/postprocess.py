from __future__ import annotations

import re
from dataclasses import replace

from m2t.models import Segment

_KANA = re.compile(r"[぀-ヿ]")
_HAN = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]{2,}")


def detect_languages(text: str) -> str:
    """依文字判斷實際語言（模型只回報單一語言，混語時不可靠）。"""
    langs = []
    if _KANA.search(text):
        langs.append("ja")
    elif _HAN.search(text):
        langs.append("zh")
    if _LATIN.search(text):
        langs.append("en")
    return "+".join(langs) or "unknown"


def to_traditional(segments: list[Segment]) -> list[Segment]:
    """含漢字且不含假名的段落轉繁體台灣用語；日文段落保持原樣。"""
    from opencc import OpenCC

    cc = OpenCC("s2twp")
    out = []
    for s in segments:
        if _KANA.search(s.text) or not _HAN.search(s.text):
            out.append(s)
        else:
            out.append(replace(s, text=cc.convert(s.text),
                               words=[replace(w, text=cc.convert(w.text)) for w in s.words]))
    return out
