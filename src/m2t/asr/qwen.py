from __future__ import annotations

from pathlib import Path

from m2t.asr.base import ASRResult
from m2t.models import Word

MODEL = "Qwen/Qwen3-ASR-1.7B"
LANG_NAMES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "yue": "Cantonese"}


def transcribe(wav: Path, language: str | None) -> ASRResult:
    from mlx_qwen3_asr import transcribe as qwen_transcribe

    lang = LANG_NAMES.get(language, language) if language else None
    result = qwen_transcribe(str(wav), model=MODEL, language=lang, return_timestamps=True, verbose=False)
    words = [Word(text=s["text"], start=float(s["start"]), end=float(s["end"])) for s in result.segments]
    return ASRResult(words=words, language=result.language)
