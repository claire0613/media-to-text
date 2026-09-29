from __future__ import annotations

import json
import re
from pathlib import Path

from m2t.merge import SENTENCE_END, join_tokens
from m2t.models import Segment, Transcript, Word

SRT_MAX = 7.0
SRT_MIN = 2.0  # 短於此的句子併入前一則字幕
_SENTENCE_RE = re.compile(r".+?(?:[。！？]+|[.!?]+(?=\s|$)|$)")


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


def _word_chunks(words: list[Word]) -> list[tuple[float, float, str]]:
    """依時間切成 ≤ SRT_MAX 秒的片段，文字由字詞組合（僅在無法對齊句子時使用）。"""
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if w.end - cur[0].start >= SRT_MAX:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return [(c[0].start, c[-1].end, join_tokens([w.text for w in c])) for c in chunks]


def _sentence_pieces(seg: Segment) -> list[tuple[float, float, str]]:
    """段落文字依句尾標點切句，時間取自對應的字詞群組。

    文字用段落文字（經整段 OpenCC，詞彙轉換正確），而非逐字重組。
    """
    sentences = [x.strip() for x in _SENTENCE_RE.findall(seg.text) if x.strip()]
    groups, cur = [], []
    for w in seg.words:
        cur.append(w)
        if w.text.strip()[-1:] in SENTENCE_END:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    if len(sentences) != len(groups):
        return _word_chunks(seg.words)

    pieces: list[tuple[float, float, str]] = []
    for text, g in zip(sentences, groups):
        start, end = g[0].start, g[-1].end
        if end - start > SRT_MAX:
            pieces += _word_chunks(g)
        elif pieces and end - pieces[-1][0] <= SRT_MIN:
            a, _, prev = pieces[-1]
            pieces[-1] = (a, end, join_tokens([prev, text]))
        else:
            pieces.append((start, end, text))
    return pieces


def render_srt(t: Transcript) -> str:
    blocks = []
    for s in t.segments:
        pieces = _sentence_pieces(s) if s.words else [(s.start, s.end, s.text)]
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
