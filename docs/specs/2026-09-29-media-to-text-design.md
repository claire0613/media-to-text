# media-to-text (m2t) 設計文件

日期：2026-09-29
參考：https://github.com/ci-yang/media-to-text-skill

## 目標

在 Apple Silicon Mac（M1 Pro / 16GB）本機、離線地把影片或錄音轉成**標註說話者**的逐字稿，支援中文（台灣）、英文、日文與混語。提供命令列工具 `m2t`，並包成 Claude Code skill `/media-to-text`，由 Claude 負責說話者命名推測、翻譯、摘要。

## 與參考專案的差異

| | 參考專案 | m2t |
|---|---|---|
| ASR | mlx-whisper large-v3-turbo | **Qwen3-ASR-1.7B**（mlx-qwen3-asr），Whisper 為備用 |
| 說話者區分 | 無（交給 LLM 猜） | **pyannote community-1** + 逐字時間戳對齊 |
| 字詞時間戳 | Whisper word_timestamps | Qwen3-ForcedAligner-0.6B |
| 輸出 | md / txt / json | md / srt / json / txt，每段含說話者 |

保留參考專案的經驗：語言與音訊一致、非中文先原文轉錄再由 Claude 翻譯、OpenCC s2twp、Whisper 設 `condition_on_previous_text=False`。

## 模型選擇理由（2026-09 調查）

- Qwen3-ASR-1.7B：開源中文最強，30 語言 + 22 中文方言（含英、日），自動語言偵測，Apache 2.0，MLX 版約 3.4GB 記憶體。
- pyannote community-1：開源 diarization 最佳（AISHELL-4 DER 11.7%），不限人數，MPS 可跑；需 HF token 並同意模型條款。
- 不選：VibeVoice-ASR（9B，16GB 記憶體跑不動）、NeMo Sortformer（最多 4 人、英文為主）、Parakeet/Canary（無中文）、FireRedASR（Mac 無 GPU 加速）。

## 架構

```
media-to-text/
├── pyproject.toml            # uv 管理，console script: m2t
├── src/m2t/
│   ├── cli.py                # 參數解析、串接流程、rename 子指令
│   ├── models.py             # 資料結構：Word, SpeakerTurn, Segment, Transcript
│   ├── fetch.py              # URL → yt-dlp；本機檔 → ffmpeg 16kHz mono WAV
│   ├── asr/
│   │   ├── base.py           # ASREngine 介面：transcribe(wav, language) -> ASRResult(words, language)
│   │   ├── qwen.py           # 預設
│   │   └── whisper.py        # --engine whisper
│   ├── diarize.py            # pyannote → list[SpeakerTurn]
│   ├── merge.py              # words + turns → segments
│   ├── postprocess.py        # OpenCC（僅中文）
│   └── export.py             # md / srt / json / txt
├── skill/SKILL.md
├── skill/templates/{general,meeting,lecture,interview,podcast}.yaml
└── tests/
```

## 資料流

1. `fetch`：輸入為 URL 時用 yt-dlp 抓音訊並取標題；本機檔取檔名為標題。一律 ffmpeg 轉 16kHz mono WAV 放在輸出資料夾。
2. `asr`：產出 `words: [{text, start, end}]` 與 `language`。
3. `diarize`（可用 `--no-diarize` 關閉）：產出 `turns: [{speaker, start, end}]`。支援 `--num-speakers`、`--min-speakers`、`--max-speakers`。pyannote 的 `SPEAKER_00` 依首次出現順序重新編號為 `SPEAKER_1`、`SPEAKER_2`…
4. `merge`：每個字分配給與其時間區間重疊最多的 turn；無重疊則取時間最近的 turn。連續同一說話者的字合併為一段；段落遇到句末標點且超過約 30 秒、或停頓 > 1.5 秒時切段。無 diarization 時所有字 speaker 為 null，只依停頓與標點切段。
5. `postprocess`：語言為中文時以 OpenCC s2twp 轉換各段文字。
6. `export`：輸出到 `output/YYYY-MM-DD_標題/`。

## 輸出格式

- `transcript.md`：標頭（來源、語言、長度、引擎、說話者清單），每段 `**[HH:MM:SS] 說話者：** 文字`
- `transcript.srt`：每段一則字幕（過長段落以句子切成 ≤ 7 秒），文字前加 `說話者：`
- `transcript.json`：`{meta, speakers: {id: name}, segments: [{speaker, start, end, text, words}]}` —— 為單一真實來源
- `transcript.txt`：`說話者：文字` 每段一行

`m2t rename <輸出資料夾> --speakers "SPEAKER_1=Claire,SPEAKER_2=Amy"`：更新 json 中的 `speakers` 對照並重新產生 md/srt/txt。首次轉錄時也可直接帶 `--speakers`。

## Skill 流程（/media-to-text）

1. 執行 `m2t`。
2. 讀 transcript.txt，依稱呼、自我介紹推測各說話者姓名，列出推測與依據（含時間點）請使用者確認，確認後執行 `m2t rename`。
3. `--bilingual`：非中文時翻譯為繁體中文，輸出 `transcript.zh-TW.md`（保留說話者與時間戳）。
4. 依 `--template` 或內容自動判斷範本，產生 `summary.md`。

## 錯誤處理

- 啟動檢查 ffmpeg（URL 時再檢查 yt-dlp），缺少時印出 `brew install` 指令並退出。
- 找不到 HF token（環境變數 `HF_TOKEN` 或 `huggingface-cli login`）或模型未授權：印出申請步驟，自動降級為不區分說話者，繼續完成。
- ASR 引擎套件未安裝：提示安裝指令並退出。

## 測試

- 單元測試（不載入模型）：merge（重疊分配、切段、無 diarization、重新編號）、postprocess、export（md/srt/json/txt 格式、時間格式）、rename。
- 整合測試（標記 `slow`，預設略過）：以 macOS `say` 合成兩種聲音的中英短音檔，跑完整流程並檢查產生兩位說話者。

## 不做（YAGNI）

聲紋庫、網頁介面、即時串流轉錄、Notion/NotebookLM 發布。
