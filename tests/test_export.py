import json

from m2t.export import fmt_ts, render_md, render_srt, render_txt, write_all
from m2t.models import Segment, Transcript, Word
from m2t.postprocess import to_traditional


def _t():
    return Transcript(
        meta={"title": "測試", "source": "a.m4a", "language": "Chinese", "duration": 65.0, "engine": "qwen"},
        speakers={"SPEAKER_1": "Claire", "SPEAKER_2": "SPEAKER_2"},
        segments=[
            Segment("SPEAKER_1", 0.0, 2.0, "大家好", [Word("大", 0, 0.5), Word("家", 0.5, 1), Word("好", 1, 2)]),
            Segment("SPEAKER_2", 62.5, 65.0, "Hi", [Word("Hi", 62.5, 65.0)]),
        ],
    )


def test_fmt_ts():
    assert fmt_ts(3725.5) == "01:02:05"
    assert fmt_ts(3725.5, srt=True) == "01:02:05,500"


def test_to_traditional():
    segs = to_traditional([Segment(None, 0, 1, "这个软件的内存", [Word("这", 0, 1)])])
    assert segs[0].text == "這個軟體的記憶體"
    assert segs[0].words[0].text == "這"


def test_render_md_uses_display_names():
    md = render_md(_t())
    assert "# 測試" in md
    assert "**[00:00:00] Claire：** 大家好" in md
    assert "**[00:01:02] SPEAKER_2：** Hi" in md


def test_render_md_without_speakers():
    t = _t()
    for s in t.segments:
        s.speaker = None
    t.speakers = {}
    assert "**[00:00:00]** 大家好" in render_md(t)


def test_render_srt():
    srt = render_srt(_t())
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,000\nClaire：大家好\n")


def test_render_srt_splits_long_segment():
    words = [Word(f"w{i}", i * 1.0, i * 1.0 + 0.9) for i in range(20)]
    t = Transcript({"title": "x"}, {}, [Segment(None, 0, 19.9, "", words)])
    blocks = render_srt(t).strip().split("\n\n")
    assert len(blocks) >= 3


def test_render_txt():
    assert render_txt(_t()) == "Claire：大家好\nSPEAKER_2：Hi\n"


def test_write_all(tmp_path):
    write_all(_t(), tmp_path)
    for name in ["transcript.json", "transcript.md", "transcript.srt", "transcript.txt"]:
        assert (tmp_path / name).exists()
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert data["speakers"]["SPEAKER_1"] == "Claire"


def test_detect_languages_from_text():
    from m2t.postprocess import detect_languages
    assert detect_languages("大家好 Hello there 好的") == "zh+en"
    assert detect_languages("今日は音声認識について話します") == "ja"
    assert detect_languages("Just English.") == "en"


def test_to_traditional_skips_japanese_segments():
    segs = to_traditional([
        Segment(None, 0, 1, "会议的软件", []),
        Segment(None, 1, 2, "会議の議事録を自動で作れる", []),
        Segment(None, 2, 3, "Hello world", []),
    ])
    assert [s.text for s in segs] == ["會議的軟體", "会議の議事録を自動で作れる", "Hello world"]
