import sys
import types
from pathlib import Path

from m2t.asr import get_engine


def test_qwen_maps_segments_to_words(monkeypatch):
    calls = {}

    def fake_transcribe(audio, **kw):
        calls.update(kw)
        return types.SimpleNamespace(
            language="Chinese", text="你好",
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


def test_attach_punctuation_from_text():
    from m2t.asr.qwen import attach_punctuation
    from m2t.models import Word

    words = [Word("Hello", 0, 1), Word("I'm", 1, 2), Word("好", 2, 3), Word("的", 3, 4), Word("那", 4, 5)]
    out = attach_punctuation(words, "Hello, I'm. 好的，那。")
    assert [w.text for w in out] == ["Hello,", "I'm.", "好", "的，", "那。"]


def test_attach_punctuation_tolerates_mismatch():
    from m2t.asr.qwen import attach_punctuation
    from m2t.models import Word

    words = [Word("abc", 0, 1), Word("xyz", 1, 2)]
    assert [w.text for w in attach_punctuation(words, "abc! zzz")] == ["abc!", "xyz"]


def _fake_qwen(monkeypatch, calls):
    def fake_transcribe(audio, **kw):
        calls.update(kw)
        return types.SimpleNamespace(language="English", text="Hi.", segments=[{"text": "Hi", "start": 0, "end": 1}])
    monkeypatch.setitem(sys.modules, "mlx_qwen3_asr", types.SimpleNamespace(transcribe=fake_transcribe))


def test_qwen_defaults_to_chinese_hint_and_auto_disables(monkeypatch):
    calls = {}
    _fake_qwen(monkeypatch, calls)
    get_engine("qwen")(Path("a.wav"), None)
    assert calls["language"] == "Chinese"
    get_engine("qwen")(Path("a.wav"), "auto")
    assert calls["language"] is None


def test_whisper_auto_means_detect(monkeypatch):
    seen = {}

    def fake_transcribe(audio, **kw):
        seen.update(kw)
        return {"language": "en", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=fake_transcribe))
    get_engine("whisper")(Path("a.wav"), "auto")
    assert seen["language"] is None
