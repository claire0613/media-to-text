# 🎙️ m2t — Media to Text

把影片或錄音轉成**標註說話者**的逐字稿，全程在 Mac 本機執行、免費、離線。支援中文（台灣）、英文、日文與混語。

```
**[00:00:03] Claire：** 大家好，今天我們討論新的專案時程。
**[00:00:09] Daniel：** I think the deadline should be next Friday.
```

## 特色

| | |
|---|---|
| 🧠 **Qwen3-ASR-1.7B** | 開源中文辨識最準的模型之一，30 語言＋22 種中文方言，中英/中日夾雜也行 |
| 👥 **說話者區分** | pyannote community-1，逐字時間戳對齊到說話者，不限人數 |
| ⚡ **Apple GPU** | MLX（ASR）＋ MPS（說話者區分） |
| 🇹🇼 **繁體台灣用語** | OpenCC s2twp（软件→軟體、内存→記憶體） |
| 📄 **多種輸出** | Markdown、SRT 字幕、JSON、純文字 |
| 🤖 **Claude Code skill** | `/media-to-text`：推測說話者姓名、翻譯、依範本摘要 |

## 安裝

需求：Apple Silicon Mac、16GB RAM 以上、Homebrew。

> ⚠️ 請把專案放在 iCloud 同步以外的資料夾（例如 `~/Developer`）。iCloud 同步的「桌面 / 文件」會讓虛擬環境失效、產生「檔名 2」副本，甚至還原 git 設定。

```bash
bash install.sh
```

會安裝 ffmpeg / yt-dlp / uv、建立環境、安裝全域指令 `m2t`，並把 skill 連結到 `~/.claude/skills/media-to-text`。

### 說話者區分需要 HuggingFace token（免費，一次性）

1. 到 [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) 登入並同意使用條款
2. 到 [Settings → Tokens](https://huggingface.co/settings/tokens) 建立 Read token
3. `uv run hf auth login`，或在 `~/.zshrc` 加 `export HF_TOKEN=hf_xxx`

沒有 token 也能用，只是不會區分說話者。

首次執行會自動下載模型（Qwen3-ASR 1.7B＋對齊模型＋pyannote，約 5GB）到 `~/.cache/huggingface/`。

## 使用

```bash
m2t meeting.m4a                                   # 自動偵測語言與人數
m2t "https://youtu.be/xxxx" --num-speakers 2      # 已知人數更準
m2t talk.mp4 --lang ja                            # 指定語言
m2t talk.mp4 --engine whisper                     # 改用 Whisper large-v3-turbo
m2t a.m4a --speakers "SPEAKER_1=Claire,SPEAKER_2=Amy"
m2t a.m4a --no-diarize                            # 只轉文字

m2t rename output/2026-09-29_週會 --speakers "SPEAKER_2=Amy"   # 事後改名
```

在 Claude Code 中：

```
/media-to-text ~/Downloads/週會.m4a --template meeting
/media-to-text https://youtu.be/xxxx --bilingual
```

## 輸出

`output/YYYY-MM-DD_標題/`

| 檔案 | 內容 |
|---|---|
| `transcript.md` | 含時間點與說話者的逐字稿 |
| `transcript.srt` | 字幕（長段落自動切成 ≤ 7 秒） |
| `transcript.json` | 完整資料（逐字時間戳、說話者對照），`rename` 以此為準 |
| `transcript.txt` | `說話者：文字`，給 LLM 用 |
| `transcript.zh-TW.md` | （skill `--bilingual`）繁中翻譯 |
| `summary.md` | （skill）依範本產生的摘要 |

摘要範本：`general`、`meeting`、`lecture`、`interview`、`podcast`（`skill/templates/`）。

## 運作方式

```
來源 ─ yt-dlp / ffmpeg → 16kHz mono WAV
     ├─ Qwen3-ASR + ForcedAligner → 每個字 (text, start, end)
     └─ pyannote community-1      → 說話者時段 (exclusive)
         ↓ 每個字分給時間重疊最多的說話者
         ↓ 同一人連續的字合併成段（換人、停頓 >1.5s、超過 30s 遇句號時切段）
         ↓ 中文 → OpenCC s2twp
         → md / srt / json / txt
```

## 模型選擇（2026-09 調查）

| 候選 | 結論 |
|---|---|
| **Qwen3-ASR-1.7B** | ✅ 預設。中文強、多語、Apache 2.0、MLX 約 3.4GB |
| Whisper large-v3-turbo | 備用（`--engine whisper`） |
| FireRedASR2 | 中文最準之一，但 Mac 無 GPU 加速 |
| Parakeet / Canary | 英文榜首，不支援中文 |
| **pyannote community-1** | ✅ 開源 diarization 最佳，AISHELL-4 DER 11.7% |
| VibeVoice-ASR | 轉錄＋說話者一次完成，但 9B，16GB 跑不動 |
| NeMo Sortformer | 最多 4 人、英文為主 |

## 開發

```bash
uv sync
uv run pytest            # 單元測試（不載模型）
uv run pytest -m slow -s # 端對端：用 macOS say 合成雙人對話實跑
```

## 授權

程式碼以 [MIT](./LICENSE) 授權。

模型不包含在本 repo 中，使用時會從 Hugging Face 下載，各自適用原本的授權：Qwen3-ASR / Qwen3-ForcedAligner（Apache 2.0）、Whisper（MIT）、pyannote community-1（CC-BY-4.0，使用時須標示出處）。
