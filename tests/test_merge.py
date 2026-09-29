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


def test_join_tokens_space_after_ascii_punctuation():
    assert join_tokens(["Hello,", "I'm", "Daniel.", "I", "think"]) == "Hello, I'm Daniel. I think"
    assert join_tokens(["Friday.", "好", "的"]) == "Friday. 好的"
    assert join_tokens(["好", "的，", "OK"]) == "好的，OK"


def _w(spec):
    """'改:1:0.0 變。:2:1.2' → words（text:speaker:start，長度 0.2 秒）"""
    out = []
    for tok in spec.split():
        text, spk, start = tok.split(":")
        out.append(Word(text, float(start), float(start) + 0.2, f"SPEAKER_{spk}"))
    return out


def test_smooth_moves_sentence_tail_back_to_previous_speaker():
    from m2t.merge import smooth_speakers
    words = _w("大:1:0.0 幅:1:0.2 的:1:0.4 改:1:0.6 變。:2:1.5 然:2:1.7 後:2:1.9")
    assert [w.speaker[-1] for w in smooth_speakers(words)] == list("1111122")


def test_smooth_keeps_real_turn_after_finished_sentence():
    from m2t.merge import smooth_speakers
    words = _w("好。:1:0.0 對。:2:0.4 然:2:0.6")
    assert [w.speaker[-1] for w in smooth_speakers(words)] == list("122")


def test_smooth_keeps_long_answer_and_big_gap():
    from m2t.merge import smooth_speakers
    words = _w("對:1:0.0 嗎？:2:3.0")  # 間隔 2.8 秒 → 是新的發言
    assert [w.speaker[-1] for w in smooth_speakers(words)] == list("12")
    words = _w("你:1:0.0 覺:1:0.2 得:1:0.4 還:2:0.8 不:2:1.0 錯。:2:1.2")  # 3 個字 → 不動
    assert [w.speaker[-1] for w in smooth_speakers(words)] == list("111222")


def test_smooth_relabels_single_char_island():
    from m2t.merge import smooth_speakers
    words = _w("練:2:0.0 習，:2:0.2 那:1:0.45 其:2:0.7 實:2:0.9")
    assert [w.speaker[-1] for w in smooth_speakers(words)] == list("22222")


def test_smooth_keeps_backchannel_with_pauses():
    from m2t.merge import smooth_speakers
    words = _w("說:2:0.0 完。:2:0.2 嗯，:1:1.0 然:2:2.0")
    assert [w.speaker[-1] for w in smooth_speakers(words)] == list("2212")


def test_smooth_does_not_mutate_input():
    from m2t.merge import smooth_speakers
    words = _w("改:1:0.0 變。:2:0.3")
    smooth_speakers(words)
    assert words[1].speaker == "SPEAKER_2"
