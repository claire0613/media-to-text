# media-to-text (m2t) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 本機把影片/錄音轉成標註說話者的逐字稿（md/srt/json/txt），並提供 `/media-to-text` Claude Code skill。

**Architecture:** 管線 fetch → asr（Qwen3-ASR 或 Whisper，逐字時間戳）→ diarize（pyannote community-1 exclusive）→ merge（字對說話者）→ postprocess（OpenCC）→ export。`transcript.json` 為單一真實來源，`m2t rename` 由它重新產生其他格式。

**Tech Stack:** Python 3.12、uv、mlx-qwen3-asr[aligner]、mlx-whisper、pyannote.audio 4.x、torch（MPS）、soundfile、opencc-python-reimplemented、ffmpeg、yt-dlp、pytest。

Spec: `docs/specs/2026-09-29-media-to-text-design.md`

## Global Constraints

- 平台：macOS Apple Silicon，16GB RAM；預設模型 `Qwen/Qwen3-ASR-1.7B`，備用 `mlx-community/whisper-large-v3-turbo`
- Diarization 模型：`pyannote/speaker-diarization-community-1`，token 讀 `HF_TOKEN` 或 huggingface 登入快取
- 說話者 ID 格式：`SPEAKER_1`、`SPEAKER_2`…（依首次出現順序）
- 輸出資料夾：`output/YYYY-MM-DD_<標題>/`
- 中文輸出一律 OpenCC `s2twp`
- 單元測試不得載入任何模型

---

### Task 1: 專案骨架、資料模型、merge

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/m2t/__init__.py`, `src/m2t/models.py`, `src/m2t/merge.py`
- Test: `tests/test_merge.py`

**Interfaces:**
- Produces:
  - `Word(text: str, start: float, end: float, speaker: str | None = None)`
  - `Turn(speaker: str, start: float, end: float)`
  - `Segment(speaker: str | None, start: float, end: float, text: str, words: list[Word])`
  - `Transcript(meta: dict, speakers: dict[str, str], segments: list[Segment])`，含 `to_dict()`、`from_dict(d)`、`display_name(speaker_id) -> str`
  - `merge.join_tokens(tokens: list[str]) -> str`
  - `merge.assign_speakers(words, turns) -> list[Word]`
  - `merge.renumber(turns) -> list[Turn]`
  - `merge.build_segments(words, max_gap=1.5, max_len=30.0) -> list[Segment]`

- [ ] **Step 1: 建立 pyproject.toml / .gitignore**

```toml
[project]
name = "m2t"
version = "0.1.0"
description = "Local media-to-text with speaker diarization (Qwen3-ASR + pyannote)"
requires-python = ">=3.10,<3.13"
dependencies = [
  "mlx-qwen3-asr[aligner]",
  "mlx-whisper",
  "pyannote.audio>=4.0",
  "soundfile",
  "opencc-python-reimplemented",
]

[project.scripts]
m2t = "m2t.cli:main"

[dependency-groups]
dev = ["pytest"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/m2t"]

[tool.pytest.ini_options]
markers = ["slow: loads real models"]
addopts = "-m 'not slow'"
```

`.gitignore`：`.venv/`、`output/`、`__pycache__/`、`*.egg-info/`、`.DS_Store`

- [ ] **Step 2: 寫失敗測試 `tests/test_merge.py`**

```python
from m2t.models import Word, Turn
from m2t.merge import join_tokens, assign_speakers, renumber, build_segments


def test_join_tokens_cjk_and_latin():
    assert join_tokens(["我", "們", "用", "Python", "寫", "code"]) == "我們用 Python 寫 code"
    assert join_tokens(["Hello", "world", "."]) == "Hello world."
    assert join_tokens(["今", "天", "，", "OK"]) == "今天，OK"


def test_renumber_by_first_appearance():
    turns = [Turn("SPEAKER_03", 0, 1), Turn("SPEAKER_00", 1, 2), Turn("SPEAKER_03", 2, 3)]
    assert [t.speaker for t in renumber(turns)] == ["SPEAKER_1", "SPEAKER_2", "SPEAKER_1"]


def test_assign_by_max_overlap_and_nearest():
    turns = [Turn("SPEAKER_1", 0.0, 2.0), Turn("SPEAKER_2", 2.0, 4.0)]
    words = [Word("a", 0.5, 1.0), Word("b", 1.8, 2.5), Word("c", 5.0, 5.2)]
    out = assign_speakers(words, turns)
    assert [w.speaker for w in out] == ["SPEAKER_1", "SPEAKER_2", "SPEAKER_2"]


def test_assign_no_turns_leaves_none():
    out = assign_speakers([Word("a", 0, 1)], [])
    assert out[0].speaker is None


def test_build_segments_splits_on_speaker_change_and_gap():
    words = [
        Word("你", 0.0, 0.2, "SPEAKER_1"), Word("好", 0.2, 0.4, "SPEAKER_1"),
        Word("Hi", 0.5, 0.7, "SPEAKER_2"),
        Word("再", 3.0, 3.2, "SPEAKER_2"), Word("見", 3.2, 3.4, "SPEAKER_2"),
    ]
    segs = build_segments(words)
    assert [(s.speaker, s.text) for s in segs] == [
        ("SPEAKER_1", "你好"), ("SPEAKER_2", "Hi"), ("SPEAKER_2", "再見"),
    ]
    assert segs[0].start == 0.0 and segs[0].end == 0.4


def test_build_segments_splits_long_at_sentence_end():
    words = []
    t = 0.0
    for i in range(40):
        words.append(Word("字", t, t + 0.9))
        t += 1.0
        if i == 31:
            words.append(Word("。", t, t))
    segs = build_segments(words, max_len=30.0)
    assert len(segs) == 2
    assert segs[0].text.endswith("。")


def test_build_segments_empty():
    assert build_segments([]) == []
```

- [ ] **Step 3: 執行 `uv run pytest tests/test_merge.py -v`，預期 FAIL（ImportError）**

- [ ] **Step 4: 實作 `src/m2t/models.py`**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass
class Turn:
    speaker: str
    start: float
    end: float


@dataclass
class Segment:
    speaker: str | None
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    meta: dict
    speakers: dict[str, str]
    segments: list[Segment]

    def display_name(self, speaker: str | None) -> str | None:
        if speaker is None:
            return None
        return self.speakers.get(speaker) or speaker

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "speakers": self.speakers,
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Transcript:
        segments = [
            Segment(
                speaker=s["speaker"], start=s["start"], end=s["end"], text=s["text"],
                words=[Word(**w) for w in s.get("words", [])],
            )
            for s in d["segments"]
        ]
        return cls(meta=d["meta"], speakers=d["speakers"], segments=segments)
```

- [ ] **Step 5: 實作 `src/m2t/merge.py`**

```python
from __future__ import annotations

import re
from dataclasses import replace

from m2t.models import Segment, Turn, Word

_LATIN = re.compile(r"[A-Za-z0-9]")
_NO_SPACE_BEFORE = set(".,!?;:%)]}'\"")
SENTENCE_END = set("。！？.!?")


def _needs_space(prev: str, cur: str) -> bool:
    if not prev or not cur or cur[0] in _NO_SPACE_BEFORE:
        return False
    # 只要一邊是拉丁字母/數字、另一邊不是標點，就加空白（中英之間也加，較易讀）
    prev_latin = bool(_LATIN.match(prev[-1]))
    cur_latin = bool(_LATIN.match(cur[0]))
    if prev_latin and cur_latin:
        return True
    if prev_latin != cur_latin:
        other = cur[0] if prev_latin else prev[-1]
        return other.isalnum()
    return False


def join_tokens(tokens: list[str]) -> str:
    out = ""
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if _needs_space(out, tok):
            out += " "
        out += tok
    return out


def renumber(turns: list[Turn]) -> list[Turn]:
    mapping: dict[str, str] = {}
    result = []
    for t in sorted(turns, key=lambda t: t.start):
        if t.speaker not in mapping:
            mapping[t.speaker] = f"SPEAKER_{len(mapping) + 1}"
        result.append(replace(t, speaker=mapping[t.speaker]))
    return result


def assign_speakers(words: list[Word], turns: list[Turn]) -> list[Word]:
    if not turns:
        return [replace(w, speaker=None) for w in words]
    out = []
    for w in words:
        best, best_overlap = None, 0.0
        for t in turns:
            overlap = min(w.end, t.end) - max(w.start, t.start)
            if overlap > best_overlap:
                best, best_overlap = t, overlap
        if best is None:
            mid = (w.start + w.end) / 2
            best = min(turns, key=lambda t: 0 if t.start <= mid <= t.end
                       else min(abs(mid - t.start), abs(mid - t.end)))
        out.append(replace(w, speaker=best.speaker))
    return out


def build_segments(words: list[Word], max_gap: float = 1.5, max_len: float = 30.0) -> list[Segment]:
    segments: list[Segment] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            segments.append(Segment(
                speaker=cur[0].speaker, start=cur[0].start, end=cur[-1].end,
                text=join_tokens([w.text for w in cur]), words=list(cur),
            ))
            cur.clear()

    for w in words:
        if cur:
            prev = cur[-1]
            speaker_changed = w.speaker != prev.speaker
            long_gap = w.start - prev.end > max_gap
            too_long = (prev.end - cur[0].start) >= max_len and prev.text.strip()[-1:] in SENTENCE_END
            if speaker_changed or long_gap or too_long:
                flush()
        cur.append(w)
    flush()
    return segments
```

`src/m2t/__init__.py`：`__version__ = "0.1.0"`

- [ ] **Step 6: `uv sync && uv run pytest tests/test_merge.py -v`，預期全部 PASS**

- [ ] **Step 7: Commit** `git add -A && git commit -m "feat: project scaffold, data models and speaker merge"`

---

### Task 2: postprocess 與 export

**Files:**
- Create: `src/m2t/postprocess.py`, `src/m2t/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `Transcript`, `Segment`, `Word`（Task 1）
- Produces:
  - `postprocess.is_chinese(language: str) -> bool`
  - `postprocess.to_traditional(segments: list[Segment]) -> list[Segment]`
  - `export.fmt_ts(sec: float, srt: bool = False) -> str`
  - `export.write_all(t: Transcript, out_dir: Path) -> None`（寫 transcript.json/md/srt/txt）
  - `export.render_md(t) -> str`、`render_srt(t) -> str`、`render_txt(t) -> str`

- [ ] **Step 1: 寫失敗測試 `tests/test_export.py`**

```python
import json

from m2t.export import fmt_ts, render_md, render_srt, render_txt, write_all
from m2t.models import Segment, Transcript, Word
from m2t.postprocess import is_chinese, to_traditional


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


def test_is_chinese():
    assert is_chinese("Chinese") and is_chinese("zh") and is_chinese("Cantonese")
    assert not is_chinese("English") and not is_chinese("ja")


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
```

- [ ] **Step 2: `uv run pytest tests/test_export.py -v`，預期 FAIL**

- [ ] **Step 3: 實作 `src/m2t/postprocess.py`**

```python
from __future__ import annotations

from dataclasses import replace

from m2t.models import Segment

_CHINESE = {"zh", "chinese", "cantonese", "mandarin", "yue"}


def is_chinese(language: str | None) -> bool:
    return bool(language) and language.lower() in _CHINESE


def to_traditional(segments: list[Segment]) -> list[Segment]:
    from opencc import OpenCC

    cc = OpenCC("s2twp")
    return [
        replace(s, text=cc.convert(s.text), words=[replace(w, text=cc.convert(w.text)) for w in s.words])
        for s in segments
    ]
```

- [ ] **Step 4: 實作 `src/m2t/export.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

from m2t.merge import SENTENCE_END, join_tokens
from m2t.models import Transcript, Word

SRT_MAX = 7.0


def fmt_ts(sec: float, srt: bool = False) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else f"{h:02d}:{m:02d}:{s:02d}"


def _prefix(t: Transcript, speaker: str | None, sep: str = "：") -> str:
    name = t.display_name(speaker)
    return f"{name}{sep}" if name else ""


def render_md(t: Transcript) -> str:
    m = t.meta
    lines = [f"# {m.get('title', 'Transcript')}", ""]
    for key, label in [("source", "來源"), ("language", "語言"), ("engine", "引擎")]:
        if m.get(key):
            lines.append(f"- {label}：{m[key]}")
    if m.get("duration") is not None:
        lines.append(f"- 長度：{fmt_ts(m['duration'])}")
    if t.speakers:
        lines.append("- 說話者：" + "、".join(t.display_name(s) for s in t.speakers))
    lines += ["", "---", ""]
    for s in t.segments:
        name = t.display_name(s.speaker)
        head = f"**[{fmt_ts(s.start)}] {name}：**" if name else f"**[{fmt_ts(s.start)}]**"
        lines += [f"{head} {s.text}", ""]
    return "\n".join(lines)


def _srt_chunks(words: list[Word]) -> list[list[Word]]:
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        long_enough = w.end - cur[0].start >= SRT_MAX
        if long_enough or (w.text.strip()[-1:] in SENTENCE_END and w.end - cur[0].start >= 2.0):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def render_srt(t: Transcript) -> str:
    blocks = []
    for s in t.segments:
        chunks = _srt_chunks(s.words) if s.words else []
        pieces = [(c[0].start, c[-1].end, join_tokens([w.text for w in c])) for c in chunks] \
            if len(chunks) > 1 else [(s.start, s.end, s.text)]
        for start, end, text in pieces:
            blocks.append((start, end, _prefix(t, s.speaker) + text))
    return "".join(
        f"{i}\n{fmt_ts(a, True)} --> {fmt_ts(b, True)}\n{text}\n\n"
        for i, (a, b, text) in enumerate(blocks, 1)
    )


def render_txt(t: Transcript) -> str:
    return "".join(f"{_prefix(t, s.speaker)}{s.text}\n" for s in t.segments)


def write_all(t: Transcript, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.json").write_text(
        json.dumps(t.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "transcript.md").write_text(render_md(t), encoding="utf-8")
    (out_dir / "transcript.srt").write_text(render_srt(t), encoding="utf-8")
    (out_dir / "transcript.txt").write_text(render_txt(t), encoding="utf-8")
```

- [ ] **Step 5: `uv run pytest -v`，預期全部 PASS**

- [ ] **Step 6: Commit** `git commit -am "feat: OpenCC postprocess and md/srt/json/txt export"`（先 `git add -A`）

---

### Task 3: fetch（ffmpeg / yt-dlp）

**Files:**
- Create: `src/m2t/fetch.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Produces:
  - `fetch.is_url(s: str) -> bool`
  - `fetch.slugify(title: str) -> str`
  - `fetch.check_tools(need_ytdlp: bool) -> None`（缺少時 `raise SystemExit` 並印 brew 指令）
  - `fetch.prepare_audio(source: str, work_dir: Path) -> tuple[Path, str, float]` → (wav 路徑, 標題, 秒數)

- [ ] **Step 1: 寫失敗測試 `tests/test_fetch.py`**

```python
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
```

- [ ] **Step 2: `uv run pytest tests/test_fetch.py -v`，預期 FAIL**

- [ ] **Step 3: 實作 `src/m2t/fetch.py`**

```python
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
```

- [ ] **Step 4: `uv run pytest tests/test_fetch.py -v`，預期 PASS**

- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: fetch/convert media to 16kHz wav"`

---

### Task 4: ASR 引擎

**Files:**
- Create: `src/m2t/asr/__init__.py`, `src/m2t/asr/base.py`, `src/m2t/asr/qwen.py`, `src/m2t/asr/whisper.py`
- Test: `tests/test_asr.py`

**Interfaces:**
- Consumes: `Word`
- Produces:
  - `asr.base.ASRResult(words: list[Word], language: str)`
  - `asr.get_engine(name: str) -> Callable[[Path, str | None], ASRResult]`（name ∈ {"qwen", "whisper"}）
  - `asr.qwen.transcribe(wav: Path, language: str | None) -> ASRResult`
  - `asr.whisper.transcribe(wav: Path, language: str | None) -> ASRResult`
  - `asr.qwen.LANG_NAMES: dict[str, str]`（`zh→Chinese, en→English, ja→Japanese, ko→Korean, yue→Cantonese`）

- [ ] **Step 1: 寫失敗測試 `tests/test_asr.py`**（以 monkeypatch 假模組，不載模型）

```python
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
```

- [ ] **Step 2: `uv run pytest tests/test_asr.py -v`，預期 FAIL**

- [ ] **Step 3: 實作**

`src/m2t/asr/base.py`
```python
from __future__ import annotations

from dataclasses import dataclass

from m2t.models import Word


@dataclass
class ASRResult:
    words: list[Word]
    language: str
```

`src/m2t/asr/__init__.py`
```python
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
```

`src/m2t/asr/qwen.py`
```python
from __future__ import annotations

from pathlib import Path

from m2t.asr.base import ASRResult
from m2t.models import Word

MODEL = "Qwen/Qwen3-ASR-1.7B"
LANG_NAMES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "yue": "Cantonese"}


def transcribe(wav: Path, language: str | None) -> ASRResult:
    from mlx_qwen3_asr import transcribe as qwen_transcribe

    lang = LANG_NAMES.get(language, language) if language else None
    result = qwen_transcribe(str(wav), model=MODEL, language=lang, return_timestamps=True, verbose=False)
    words = [Word(text=s["text"], start=float(s["start"]), end=float(s["end"])) for s in result.segments]
    return ASRResult(words=words, language=result.language)
```

`src/m2t/asr/whisper.py`
```python
from __future__ import annotations

from pathlib import Path

from m2t.asr.base import ASRResult
from m2t.models import Word

MODEL = "mlx-community/whisper-large-v3-turbo"
PROMPTS = {"zh": "以下是繁體中文的錄音，可能包含英文術語。"}


def transcribe(wav: Path, language: str | None) -> ASRResult:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(wav), path_or_hf_repo=MODEL, language=language,
        initial_prompt=PROMPTS.get(language or ""),
        condition_on_previous_text=False, word_timestamps=True, verbose=False,
    )
    words = [
        Word(text=w["word"].strip(), start=float(w["start"]), end=float(w["end"]))
        for seg in result["segments"] for w in seg.get("words", [])
    ]
    return ASRResult(words=words, language=result["language"])
```

- [ ] **Step 4: `uv run pytest tests/test_asr.py -v`，預期 PASS**

- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: Qwen3-ASR and Whisper engines"`

---

### Task 5: diarize

**Files:**
- Create: `src/m2t/diarize.py`
- Test: `tests/test_diarize.py`

**Interfaces:**
- Consumes: `Turn`, `merge.renumber`
- Produces:
  - `diarize.get_token() -> str | None`
  - `diarize.diarize(wav: Path, num_speakers=None, min_speakers=None, max_speakers=None) -> list[Turn] | None`（無 token 或載入失敗回 None 並印提示）

- [ ] **Step 1: 寫失敗測試 `tests/test_diarize.py`**

```python
from m2t import diarize


def test_no_token_returns_none(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(diarize, "get_token", lambda: None)
    assert diarize.diarize(tmp_path / "a.wav") is None
    assert "HF_TOKEN" in capsys.readouterr().err


def test_get_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert diarize.get_token() == "hf_x"
```

- [ ] **Step 2: 執行，預期 FAIL**

- [ ] **Step 3: 實作 `src/m2t/diarize.py`**

```python
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
```

注意：pyannote 4 的 `for turn, speaker in output.speaker_diarization` 迭代介面同時適用 exclusive。

- [ ] **Step 4: 執行，預期 PASS**

- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: pyannote community-1 diarization with graceful fallback"`

---

### Task 6: CLI（含 rename）

**Files:**
- Create: `src/m2t/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: 上述所有
- Produces:
  - `cli.parse_speakers(s: str | None) -> dict[str, str]`
  - `cli.run(args) -> Path`（回傳輸出資料夾）
  - `cli.rename(out_dir: Path, mapping: dict[str, str]) -> None`
  - `cli.main(argv: list[str] | None = None) -> None`
  - 指令：`m2t <source> [-o DIR] [--engine qwen|whisper] [--lang CODE] [--num-speakers N] [--min-speakers N] [--max-speakers N] [--no-diarize] [--speakers MAP]`；`m2t rename <dir> --speakers MAP`

- [ ] **Step 1: 寫失敗測試 `tests/test_cli.py`**

```python
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
```

- [ ] **Step 2: 執行，預期 FAIL**

- [ ] **Step 3: 實作 `src/m2t/cli.py`**

```python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path

from m2t.asr import get_engine
from m2t.diarize import diarize
from m2t.export import write_all
from m2t.fetch import check_tools, is_url, prepare_audio, slugify
from m2t.merge import assign_speakers, build_segments
from m2t.models import Transcript
from m2t.postprocess import is_chinese, to_traditional


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_speakers(s: str | None) -> dict[str, str]:
    if not s:
        return {}
    pairs = (p.split("=", 1) for p in s.split(",") if "=" in p)
    return {k.strip(): v.strip() for k, v in pairs}


def run(args: argparse.Namespace) -> Path:
    check_tools(need_ytdlp=is_url(args.source))
    tmp_dir = Path(args.output or "output") / ".work"
    started = time.time()

    log("🎬 準備音訊…")
    wav, title, duration = prepare_audio(args.source, tmp_dir)
    out_dir = Path(args.output) if args.output else Path("output") / f"{date.today():%Y-%m-%d}_{slugify(title)}"

    log(f"📝 轉錄中（{args.engine}）…")
    asr = get_engine(args.engine)(wav, args.lang)
    log(f"   語言：{asr.language}，{len(asr.words)} 個字詞")

    turns = None
    if not args.no_diarize:
        log("👥 區分說話者…")
        turns = diarize(wav, args.num_speakers, args.min_speakers, args.max_speakers)

    words = assign_speakers(asr.words, turns or [])
    segments = build_segments(words)
    if is_chinese(asr.language):
        segments = to_traditional(segments)

    speaker_ids = list(dict.fromkeys(s.speaker for s in segments if s.speaker))
    names = parse_speakers(args.speakers)
    transcript = Transcript(
        meta={"title": title, "source": args.source, "language": asr.language,
              "duration": round(duration, 2), "engine": args.engine, "diarized": turns is not None},
        speakers={sid: names.get(sid, sid) for sid in speaker_ids},
        segments=segments,
    )
    write_all(transcript, out_dir)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    log(f"✅ 完成，耗時 {time.time() - started:.0f} 秒（音訊 {duration:.0f} 秒），"
        f"{len(speaker_ids)} 位說話者 → {out_dir}")
    return out_dir


def rename(out_dir: Path, mapping: dict[str, str]) -> None:
    path = out_dir / "transcript.json"
    t = Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
    unknown = set(mapping) - set(t.speakers)
    if unknown:
        raise SystemExit(f"❌ 找不到說話者：{', '.join(sorted(unknown))}（可用：{', '.join(t.speakers)}）")
    t.speakers.update(mapping)
    write_all(t, out_dir)
    log(f"✅ 已更新說話者名稱 → {out_dir}")


def main(argv: list[str] | None = None) -> Path:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "rename":
        p = argparse.ArgumentParser(prog="m2t rename")
        p.add_argument("dir")
        p.add_argument("--speakers", required=True, help='例如 "SPEAKER_1=Claire,SPEAKER_2=Amy"')
        a = p.parse_args(argv[1:])
        rename(Path(a.dir), parse_speakers(a.speakers))
        return Path(a.dir)

    p = argparse.ArgumentParser(prog="m2t", description="影片/錄音 → 標註說話者的逐字稿")
    p.add_argument("source", help="本機檔案或網址（YouTube 等）")
    p.add_argument("-o", "--output", help="輸出資料夾（預設 output/日期_標題）")
    p.add_argument("--engine", choices=["qwen", "whisper"], default="qwen")
    p.add_argument("--lang", help="語言代碼 zh/en/ja…（預設自動偵測）")
    p.add_argument("--num-speakers", type=int)
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--no-diarize", action="store_true", help="不區分說話者")
    p.add_argument("--speakers", help='說話者命名，例如 "SPEAKER_1=Claire"')
    return run(p.parse_args(argv))
```

- [ ] **Step 4: `uv run pytest -v`，預期全部 PASS**

- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: m2t CLI with rename subcommand"`

---

### Task 7: 整合測試與實機驗證

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 寫 slow 整合測試**

```python
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
```

- [ ] **Step 2: 執行 `uv run pytest -m slow -v -s`**（需 HF_TOKEN；首次會下載模型約 5GB）。預期 PASS；若 Qwen 的 `segments` 結構與文件不同，依實際輸出修正 `asr/qwen.py` 並補單元測試。

- [ ] **Step 3: Commit** `git add -A && git commit -m "test: slow end-to-end integration test"`

---

### Task 8: Skill、範本、README

**Files:**
- Create: `skill/SKILL.md`, `skill/templates/{general,meeting,lecture,interview,podcast}.yaml`, `README.md`, `install.sh`

- [ ] **Step 1: `install.sh`**：檢查/安裝 brew 套件（ffmpeg yt-dlp uv）→ `uv sync` → `uv tool install --editable .`（全域 `m2t` 指令）→ 將 `skill/` symlink 到 `~/.claude/skills/media-to-text` → 提示 HF token 步驟。

- [ ] **Step 2: `skill/SKILL.md`**：frontmatter `name: media-to-text`、description；流程：
  1. 執行 `m2t "<source>" [--num-speakers N]`，取得輸出資料夾
  2. 讀 `transcript.txt`，推測說話者姓名，以表格列出「ID → 名字 → 依據（時間點＋原句）」，請使用者確認後執行 `m2t rename <dir> --speakers ...`
  3. `--bilingual` 且非中文：翻譯為繁中 `transcript.zh-TW.md`，保留時間戳與說話者，專有名詞保留原文
  4. 依 `--template` 或內容關鍵字選範本，讀 `templates/<id>.yaml` 依 sections 產生 `summary.md`（繁中）

- [ ] **Step 3: 範本** yaml 結構：`id, name, keywords, system_prompt, sections[{id, title, prompt}]`。general（TLDR/重點/摘要）、meeting（出席者/議程/決策/待辦）、lecture（大綱/概念/金句/收穫）、interview（問答/評估）、podcast（見解/金句/摘要）。

- [ ] **Step 4: README.md**（繁中）：功能、安裝、HF token 步驟、用法範例、輸出格式、模型選擇理由、效能。

- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: Claude Code skill, summary templates, README, installer"`
