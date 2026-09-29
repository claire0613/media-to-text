from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s))


def slugify(title: str) -> str:
    s = title.replace("/", "-")
    s = re.sub(r"[\\:*?\"<>|]", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:80] or "untitled"


def check_tools(need_ytdlp: bool) -> None:
    missing = [t for t in ["ffmpeg"] + (["yt-dlp"] if need_ytdlp else []) if not shutil.which(t)]
    if missing:
        print(f"❌ 缺少工具：{', '.join(missing)}\n   請執行：brew install {' '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)


def _download(url: str, work_dir: Path) -> tuple[Path, str]:
    title = subprocess.run(
        ["yt-dlp", "--print", "title", "--no-playlist", url],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    subprocess.run(
        ["yt-dlp", "-f", "bestaudio/best", "--no-playlist", "-o", str(work_dir / "source.%(ext)s"), url],
        check=True,
    )
    return next(work_dir.glob("source.*")), title


def prepare_audio(source: str, work_dir: Path) -> tuple[Path, str, float]:
    work_dir.mkdir(parents=True, exist_ok=True)
    if is_url(source):
        src, title = _download(source, work_dir)
    else:
        src = Path(source).expanduser().resolve()
        if not src.exists():
            raise SystemExit(f"❌ 找不到檔案：{src}")
        title = src.stem
    wav = work_dir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
    )
    import soundfile as sf

    return wav, title, sf.info(str(wav)).duration
