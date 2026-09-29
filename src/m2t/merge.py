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
