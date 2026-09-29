import subprocess
from pathlib import Path

import pytest

from m2t import cli


@pytest.mark.slow
def test_two_voices(tmp_path):
    parts = []
    for i, (voice, text) in enumerate([
        ("Meijia", "大家好，我是美佳，今天我們討論新的專案時程。"),
        ("Daniel", "Hello, I'm Daniel. I think the deadline should be next Friday."),
        ("Meijia", "好的，那我們就訂在下週五。"),
    ]):
        f = tmp_path / f"{i}.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(f), text], check=True)
        parts.append(f)
    lst = tmp_path / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    src = tmp_path / "dialog.m4a"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), str(src)], check=True)
    out = cli.main([str(src), "-o", str(tmp_path / "out"), "--num-speakers", "2"])
    txt = (Path(out) / "transcript.txt").read_text(encoding="utf-8")
    assert "SPEAKER_1" in txt and "SPEAKER_2" in txt
    assert "專案" in txt and "Friday" in txt
