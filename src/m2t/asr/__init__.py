from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from m2t.asr.base import ASRResult


def get_engine(name: str) -> Callable[[Path, str | None], ASRResult]:
    if name == "qwen":
        from m2t.asr.qwen import transcribe
    elif name == "whisper":
        from m2t.asr.whisper import transcribe
    else:
        raise ValueError(f"unknown engine: {name}")
    return transcribe
