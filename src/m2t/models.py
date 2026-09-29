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
