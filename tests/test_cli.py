import json
from pathlib import Path

from m2t import cli
from m2t.asr.base import ASRResult
from m2t.models import Turn, Word


def test_parse_speakers():
    assert cli.parse_speakers("SPEAKER_1=Claire, SPEAKER_2=Amy") == {"SPEAKER_1": "Claire", "SPEAKER_2": "Amy"}
    assert cli.parse_speakers(None) == {}


def _fake_pipeline(monkeypatch, tmp_path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(cli, "check_tools", lambda need_ytdlp: None)
    monkeypatch.setattr(cli, "prepare_audio", lambda src, work: (wav, "會議", 4.0))
    monkeypatch.setattr(cli, "get_engine", lambda name: lambda w, lang: ASRResult(
        [Word("软", 0, 0.5), Word("件", 0.5, 1), Word("OK", 2.5, 3)], "Chinese"))
    monkeypatch.setattr(cli, "diarize", lambda *a, **k: [Turn("SPEAKER_1", 0, 2), Turn("SPEAKER_2", 2, 4)])


def test_run_end_to_end_with_fakes(monkeypatch, tmp_path):
    _fake_pipeline(monkeypatch, tmp_path)
    out = cli.main(["x.m4a", "-o", str(tmp_path / "out"), "--speakers", "SPEAKER_1=Claire"])
    data = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
    assert data["speakers"] == {"SPEAKER_1": "Claire", "SPEAKER_2": "SPEAKER_2"}
    assert [s["text"] for s in data["segments"]] == ["軟體", "OK"]
    assert "Claire：軟體" in (out / "transcript.txt").read_text(encoding="utf-8")
    assert not (out / "audio.wav").exists()


def test_rename(monkeypatch, tmp_path):
    _fake_pipeline(monkeypatch, tmp_path)
    out = cli.main(["x.m4a", "-o", str(tmp_path / "out")])
    cli.main(["rename", str(out), "--speakers", "SPEAKER_2=Amy"])
    assert "Amy：OK" in (out / "transcript.txt").read_text(encoding="utf-8")
    assert "**[00:00:02] Amy：** OK" in (out / "transcript.md").read_text(encoding="utf-8")
