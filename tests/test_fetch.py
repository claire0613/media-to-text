import subprocess

from m2t.fetch import is_url, prepare_audio, slugify


def test_is_url():
    assert is_url("https://youtu.be/abc")
    assert not is_url("/Users/me/a.m4a")


def test_slugify():
    assert slugify('團隊 週會: 9/29 "final"') == "團隊_週會_9-29_final"
    assert len(slugify("a" * 200)) <= 80


def test_prepare_local_audio(tmp_path):
    src = tmp_path / "tone.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=2", str(src)], check=True)
    wav, title, duration = prepare_audio(str(src), tmp_path / "work")
    assert wav.exists() and wav.suffix == ".wav"
    assert title == "tone"
    assert 1.9 < duration < 2.2
    import soundfile as sf
    info = sf.info(str(wav))
    assert info.samplerate == 16000 and info.channels == 1
