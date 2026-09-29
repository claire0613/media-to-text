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
