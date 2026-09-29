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
uv sync

echo "🔧 安裝全域指令 m2t…（改程式後重跑本腳本以更新）"
uv tool install --force --python 3.12 .   # 非 editable：避免 macOS hidden .pth 被 Python 略過

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

echo "✅ 完成！試試：m2t ~/Downloads/meeting.m4a"
