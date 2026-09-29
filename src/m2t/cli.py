from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import date
from dataclasses import replace
from pathlib import Path

from m2t.asr import get_engine
from m2t.diarize import diarize
from m2t.export import write_all
from m2t.fetch import check_tools, is_url, prepare_audio, slugify
from m2t.merge import assign_speakers, build_segments, smooth_speakers
from m2t.models import Transcript, Word
from m2t.postprocess import detect_languages, to_traditional


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_mapping(s: str | None) -> dict[str, str]:
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
    log(f"   {len(asr.words)} 個字詞")

    turns = None
    if not args.no_diarize:
        log("👥 區分說話者…")
        turns = diarize(wav, args.num_speakers, args.min_speakers, args.max_speakers)

    words = smooth_speakers(assign_speakers(asr.words, turns or []))
    segments = build_segments(words)
    segments = to_traditional(segments)
    language = detect_languages(" ".join(s.text for s in segments))

    speaker_ids = list(dict.fromkeys(s.speaker for s in segments if s.speaker))
    names = parse_mapping(args.speakers)
    transcript = Transcript(
        meta={"title": title, "source": args.source, "language": language,
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


def replace_in_words(words: list[Word], mapping: dict[str, str]) -> list[Word]:
    """在逐字資料接成的字串上取代，結果歸給比對起點所在的字（可跨字、可改變長度）。"""
    joined = "".join(w.text for w in words)
    owner = [i for i, w in enumerate(words) for _ in w.text]
    texts = [""] * len(words)
    pattern = re.compile("|".join(sorted(map(re.escape, mapping), key=len, reverse=True)))
    pos = 0
    for m in pattern.finditer(joined):
        for k in range(pos, m.start()):
            texts[owner[k]] += joined[k]
        new = mapping[m.group()]
        if len(new) == len(m.group()):  # 等長：逐字放回原本的字，保留各字時間戳
            for k, ch in enumerate(new):
                texts[owner[m.start() + k]] += ch
        else:
            texts[owner[m.start()]] += new
        pos = m.end()
    for k in range(pos, len(joined)):
        texts[owner[k]] += joined[k]
    return [replace(w, text=t) for w, t in zip(words, texts)]


def fix(out_dir: Path, mapping: dict[str, str]) -> None:
    """以「錯=對」批次修正辨識錯字，段落文字與逐字資料（SRT 長句會用到）一起改。"""
    path = out_dir / "transcript.json"
    t = Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
    counts = {k: sum(s.text.count(k) for s in t.segments) for k in mapping}
    for s in t.segments:
        for wrong, right in mapping.items():
            s.text = s.text.replace(wrong, right)
        s.words = replace_in_words(s.words, mapping)
    t.meta["fixes"] = {**t.meta.get("fixes", {}), **mapping}
    write_all(t, out_dir)
    for k, n in counts.items():
        log(f"   {k} → {mapping[k]}：{n} 處" if n else f"⚠️  找不到「{k}」")
    log(f"✅ 已修正 → {out_dir}")


def main(argv: list[str] | None = None) -> Path:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "fix":
        p = argparse.ArgumentParser(prog="m2t fix")
        p.add_argument("dir")
        p.add_argument("--replace", required=True, help='例如 "明強=冥想,殺生=發聲"')
        a = p.parse_args(argv[1:])
        fix(Path(a.dir), parse_mapping(a.replace))
        return Path(a.dir)
    if argv and argv[0] == "rename":
        p = argparse.ArgumentParser(prog="m2t rename")
        p.add_argument("dir")
        p.add_argument("--speakers", required=True, help='例如 "SPEAKER_1=Claire,SPEAKER_2=Amy"')
        a = p.parse_args(argv[1:])
        rename(Path(a.dir), parse_mapping(a.speakers))
        return Path(a.dir)

    p = argparse.ArgumentParser(prog="m2t", description="影片/錄音 → 標註說話者的逐字稿")
    p.add_argument("source", help="本機檔案或網址（YouTube 等）")
    p.add_argument("-o", "--output", help="輸出資料夾（預設 output/日期_標題）")
    p.add_argument("--engine", choices=["qwen", "whisper"], default="qwen")
    p.add_argument("--lang", help="語言提示 zh/en/ja/auto（qwen 預設 zh 提示，不會翻譯其他語言；whisper 預設自動偵測）")
    p.add_argument("--num-speakers", type=int)
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--no-diarize", action="store_true", help="不區分說話者")
    p.add_argument("--speakers", help='說話者命名，例如 "SPEAKER_1=Claire"')
    return run(p.parse_args(argv))


def entry() -> None:
    """console script 入口：main() 的回傳值會被 sys.exit 當成錯誤，這裡丟棄它。"""
    main()
