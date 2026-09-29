#!/usr/bin/env bash
# 安裝 m2t：系統工具、Python 環境、全域指令、Claude Code skill
set -euo pipefail
cd "$(dirname "$0")"

missing=()
for tool in ffmpeg yt-dlp uv; do
  command -v "$tool" >/dev/null || missing+=("$tool")
done
if ((${#missing[@]})); then
  echo "📦 安裝 ${missing[*]}…"
  brew install "${missing[@]}"
fi

echo "🐍 建立 Python 環境…"
# iCloud 同步的資料夾（Desktop / Documents）會把 .venv 內檔案標成 hidden（Python 會略過 hidden 的 .pth）、
# 產生「檔名 2」衝突副本、甚至還原 .git/config。請把專案放在 iCloud 以外的位置，例如 ~/Developer。
case "$PWD" in
  "$HOME/Desktop"*|"$HOME/Documents"*|"$HOME/Library/Mobile Documents"*)
    if xattr "$HOME/Desktop" 2>/dev/null | grep -q com.apple.icloud; then
      echo "⚠️  專案位於 iCloud 同步資料夾（$PWD），建議搬到 ~/Developer 後再安裝。" >&2
    fi ;;
esac
uv sync

echo "🔧 安裝全域指令 m2t…"
uv tool install --force --python 3.12 --editable .
uv tool update-shell >/dev/null 2>&1 || true   # 確保 ~/.local/bin 在 PATH（開新終端機後生效）

skill_dir="$HOME/.claude/skills/media-to-text"
mkdir -p "$(dirname "$skill_dir")"
ln -sfn "$PWD/skill" "$skill_dir"
echo "🤖 Claude Code skill 已連結：$skill_dir"

if [[ -z "${HF_TOKEN:-}" ]] && ! uv run python -c "import huggingface_hub,sys; sys.exit(0 if huggingface_hub.get_token() else 1)" 2>/dev/null; then
  cat <<'EOF'

⚠️  尚未設定 HuggingFace token（區分說話者需要）：
   1. 到 https://huggingface.co/pyannote/speaker-diarization-community-1 登入並同意使用條款
   2. 到 https://huggingface.co/settings/tokens 建立 Read token
   3. 執行：uv run hf auth login   （或在 ~/.zshrc 加上 export HF_TOKEN=hf_xxx）
EOF
fi

echo "✅ 完成！開一個新的終端機視窗，試試：m2t ~/Downloads/meeting.m4a"
