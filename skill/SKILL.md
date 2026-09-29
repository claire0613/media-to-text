---
name: media-to-text
description: 將影片或錄音（本機檔案或 YouTube 等網址）轉成標註說話者的逐字稿，並推測說話者姓名、可選翻譯成繁體中文、依範本產生摘要。當使用者要把影片/錄音/會議/Podcast 轉文字、做逐字稿、字幕或會議記錄時使用。
argument-hint: <檔案或網址> [--num-speakers N] [--lang zh|en|ja] [--bilingual] [--template general|meeting|lecture|interview|podcast] [--no-diarize]
---

# Media to Text

用本機 `m2t` 指令（Qwen3-ASR + pyannote，全離線）產生逐字稿，再由你完成說話者命名、翻譯與摘要。

## 參數

從使用者輸入解析：
- `<source>`：本機檔案路徑或網址（必要）
- `--num-speakers N` / `--lang` / `--no-diarize` / `--engine whisper`：直接轉給 `m2t`
- `--bilingual`：非中文內容額外產生繁體中文翻譯
- `--template <id>`：摘要範本；未指定則自動判斷

## 步驟

### 1. 轉錄

```bash
m2t "<source>" [轉給 m2t 的參數]
```

- 長音檔會跑數分鐘，用背景執行並等待完成。
- 最後一行 `✅ 完成 … → <輸出資料夾>` 即輸出位置。
- 若出現「無法進行說話者區分」，照訊息引導使用者設定 HF_TOKEN；逐字稿仍會產出（無說話者）。
- 若 `m2t` 不存在，請使用者在專案資料夾執行 `bash install.sh`。

### 2. 推測說話者姓名（有說話者時）

讀 `<輸出資料夾>/transcript.txt`，從自我介紹、互相稱呼（「謝謝 Amy」「Claire 你覺得呢」）、角色線索推測每個 `SPEAKER_N` 是誰。用表格呈現：

| ID | 推測姓名 | 依據（時間點＋原句） |
|---|---|---|

無法判斷的保留原 ID。**等使用者確認或修改後**才執行：

```bash
m2t rename "<輸出資料夾>" --speakers "SPEAKER_1=Claire,SPEAKER_2=Amy"
```

### 3. 翻譯（`--bilingual` 且語言非中文）

讀 `transcript.md`，逐段翻譯為繁體中文（台灣用語），保留 `**[時間] 說話者：**` 前綴，專有名詞與人名保留原文，寫入 `transcript.zh-TW.md`。長逐字稿分批翻譯。

### 4. 摘要

1. 範本：用 `--template`；否則讀逐字稿前 800 字，比對 `templates/*.yaml` 的 `keywords`，命中最多者勝，都沒有則用 `general`。
2. 讀 `templates/<id>.yaml`（位於本 skill 資料夾），依 `system_prompt` 的角色、按 `sections` 順序逐段產生內容。
3. 一律使用繁體中文；引用內容時標註說話者與時間點 `[00:12:34]`。
4. 寫入 `<輸出資料夾>/summary.md`，開頭放標題、日期、來源、說話者。

### 5. 回報

列出輸出資料夾與檔案、語言、長度、說話者，並附 3～5 行重點。
