from __future__ import annotations

import os
import sys
from pathlib import Path

from m2t.merge import renumber
from m2t.models import Turn

MODEL = "pyannote/speaker-diarization-community-1"
HELP = f"""⚠️  無法進行說話者區分，改為只轉文字。
   1. 到 https://huggingface.co/{MODEL} 登入並同意使用條款
   2. 到 https://huggingface.co/settings/tokens 建立 Read token
   3. export HF_TOKEN=hf_xxx（或執行 uv run hf auth login）"""


def get_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token as hf_get_token
        return hf_get_token()
    except Exception:
        return None


def diarize(wav: Path, num_speakers: int | None = None, min_speakers: int | None = None,
            max_speakers: int | None = None) -> list[Turn] | None:
    token = get_token()
    if not token:
        print(HELP.replace("無法", "找不到 HF_TOKEN，無法"), file=sys.stderr)
        return None
    try:
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(MODEL, token=token)
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
    except Exception as e:  # 未授權、網路錯誤等
        print(f"{HELP}\n   錯誤：{e}", file=sys.stderr)
        return None

    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    kwargs = {k: v for k, v in
              {"num_speakers": num_speakers, "min_speakers": min_speakers, "max_speakers": max_speakers}.items()
              if v is not None}
    output = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
    annotation = getattr(output, "exclusive_speaker_diarization", None) or output.speaker_diarization
    turns = [Turn(str(spk), float(seg.start), float(seg.end)) for seg, spk in annotation]
    return renumber(turns)
