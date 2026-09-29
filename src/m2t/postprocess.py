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


# 不用 s2twp：它的詞彙轉換會把日常用語當 IT 術語（對象→物件、類型→型別、連接→連線）。
# s2tw 只轉字形，但「只/隻」「發/髮」一對多仍會猜錯，以下修正實際錄音中出現過的誤轉。
_FIXES = {
    "隻有": "只有", "隻要": "只要", "隻是": "只是", "隻能": "只能", "隻好": "只好",
    "隻會": "只會", "隻剩": "只剩", "不隻": "不只", "隻不過": "只不過",
    "髮展": "發展", "髮生": "發生", "髮現": "發現",
    "賬號": "帳號", "賬戶": "帳戶",  # s2tw 用「賬」，台灣慣用「帳」
}
_FIX_RE = re.compile("|".join(sorted(map(re.escape, _FIXES), key=len, reverse=True)))


def _converter():
    from opencc import OpenCC

    cc = OpenCC("s2tw")
    return lambda text: _FIX_RE.sub(lambda m: _FIXES[m.group()], cc.convert(text))


def to_traditional(segments: list[Segment]) -> list[Segment]:
    """含漢字且不含假名的段落轉繁體；日文段落保持原樣。"""
    convert = _converter()
    out = []
    for s in segments:
        if _KANA.search(s.text) or not _HAN.search(s.text):
            out.append(s)
        else:
            out.append(replace(s, text=convert(s.text), words=_convert_words(s.words, convert)))
    return out


def _convert_words(words, convert):
    """整串一起轉（有上下文，「关|系」才會變「關係」），再依原長度切回每個字。"""
    joined = "".join(w.text for w in words)
    converted = convert(joined)
    if len(converted) != len(joined):
        return [replace(w, text=convert(w.text)) for w in words]
    out, pos = [], 0
    for w in words:
        out.append(replace(w, text=converted[pos:pos + len(w.text)]))
        pos += len(w.text)
    return out
