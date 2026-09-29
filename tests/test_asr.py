import sys
import types
from pathlib import Path

from m2t.asr import get_engine


def test_qwen_maps_segments_to_words(monkeypatch):
    calls = {}

    def fake_transcribe(audio, **kw):
        calls.update(kw)
        return types.SimpleNamespace(
            language="Chinese",
            segments=[{"text": "你", "start": 0.0, "end": 0.2}, {"text": "好", "start": 0.2, "end": 0.4}],
        )

    monkeypatch.setitem(sys.modules, "mlx_qwen3_asr", types.SimpleNamespace(transcribe=fake_transcribe))
    res = get_engine("qwen")(Path("a.wav"), "zh")
    assert res.language == "Chinese"
    assert [w.text for w in res.words] == ["你", "好"]
    assert calls["language"] == "Chinese" and calls["return_timestamps"] is True
    assert calls["model"] == "Qwen/Qwen3-ASR-1.7B"


def test_whisper_flattens_words(monkeypatch):
    def fake_transcribe(audio, **kw):
        assert kw["condition_on_previous_text"] is False
        return {"language": "en", "segments": [
            {"words": [{"word": " Hello", "start": 0.0, "end": 0.4}, {"word": " world", "start": 0.4, "end": 0.8}]},
        ]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=fake_transcribe))
    res = get_engine("whisper")(Path("a.wav"), None)
    assert res.language == "en"
    assert [w.text for w in res.words] == ["Hello", "world"]
