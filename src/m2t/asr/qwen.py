from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from m2t.asr.base import ASRResult
from m2t.models import Word

MODEL = "Qwen/Qwen3-ASR-1.7B"
LANG_NAMES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "yue": "Cantonese"}
# 自動偵測時，1.7B 會把整個片段判成單一語言並丟掉另一語言的句子（中英切換時會漏中文）。
# 給 Chinese 提示不會把英文/日文翻成中文（已實測），因此作為預設；--lang auto 可關閉。
DEFAULT_HINT = "Chinese"


def attach_punctuation(words: list[Word], text: str) -> list[Word]:
    """aligner 的字詞不含標點；依序在原文中找到每個字，把緊接其後的標點接回該字。"""
    out, pos = [], 0
    for w in words:
        idx = text.find(w.text, pos)
        if idx < 0:
            out.append(w)
            continue
        end = idx + len(w.text)
        tail = end
        while tail < len(text) and not text[tail].isspace() and not text[tail].isalnum():
            tail += 1
        out.append(replace(w, text=text[idx:tail]))
        pos = tail
    return out


def transcribe(wav: Path, language: str | None) -> ASRResult:
    from mlx_qwen3_asr import transcribe as qwen_transcribe

    if language == "auto":
        lang = None
    elif language:
        lang = LANG_NAMES.get(language, language)
    else:
        lang = DEFAULT_HINT
    result = qwen_transcribe(str(wav), model=MODEL, language=lang, return_timestamps=True, verbose=False)
    words = [Word(text=s["text"], start=float(s["start"]), end=float(s["end"])) for s in result.segments]
    return ASRResult(words=attach_punctuation(words, result.text), language=result.language)
