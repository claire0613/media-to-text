from m2t import diarize


def test_no_token_returns_none(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(diarize, "get_token", lambda: None)
    assert diarize.diarize(tmp_path / "a.wav") is None
    assert "HF_TOKEN" in capsys.readouterr().err


def test_get_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert diarize.get_token() == "hf_x"
