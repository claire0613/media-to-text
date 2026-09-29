from __future__ import annotations

from pathlib import Path

from m2t.asr.base import ASRResult
from m2t.models import Word

MODEL = "mlx-community/whisper-large-v3-turbo"
PROMPTS = {"zh": "以下是繁體中文的錄音，可能包含英文術語。"}


def transcribe(wav: Path, language: str | None) -> ASRResult:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(wav), path_or_hf_repo=MODEL, language=language,
        initial_prompt=PROMPTS.get(language or ""),
        condition_on_previous_text=False, word_timestamps=True, verbose=False,
    )
    words = [
        Word(text=w["word"].strip(), start=float(w["start"]), end=float(w["end"]))
        for seg in result["segments"] for w in seg.get("words", [])
    ]
    return ASRResult(words=words, language=result["language"])
