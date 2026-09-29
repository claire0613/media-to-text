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
