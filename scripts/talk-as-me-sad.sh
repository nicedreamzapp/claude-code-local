#!/bin/bash
# talk-as-me-sad — turn text into a SadTalker video of Matt using the approved config.
#
# Approved 2026-04-24: --still + --preprocess full + --enhancer gfpgan on the
# dark-blazer headshot. Other configs land in the uncanny zone for this user.
#
# Pipeline:
#   text -> Pocket TTS (Matt's voice clone) -> speech.wav
#   speech.wav + portrait -> SadTalker (--still) -> talking-head mp4
#
# Usage:
#   talk-as-me-sad.sh "your script here" [photo.png]
#   talk-as-me-sad.sh --script-file path.txt [photo.png]
#
# Default photo: ~/Desktop/Local AI Setup/scripts/avatar-source.png

set -euo pipefail

SAD_DIR="$HOME/.local-ai/SadTalker"
SCRIPTS_DIR="$HOME/Desktop/Local AI Setup/scripts"
OUTPUT_DIR="$HOME/Desktop/Local AI Setup/output"
TTS_SERVER="http://localhost:8000"
VOICE_SAMPLE="$HOME/Library/Application Support/sh.voicebox.app/voice-sample-backup.wav"
CHATTERBOX_VENV="$HOME/chatterbox-env"
DEFAULT_PHOTO="$SCRIPTS_DIR/avatar-source.png"

# Parse args
if [ "${1:-}" = "--script-file" ]; then
    [ -n "${2:-}" ] || { echo "Error: --script-file requires a path" >&2; exit 1; }
    TEXT=$(cat "$2")
    shift 2
elif [ -n "${1:-}" ]; then
    TEXT="$1"
    shift
else
    cat >&2 <<EOF
Usage: $(basename "$0") "script text" [photo.png]
       $(basename "$0") --script-file path.txt [photo.png]

Default photo: $DEFAULT_PHOTO
EOF
    exit 1
fi

PHOTO="${1:-$DEFAULT_PHOTO}"
[ -f "$PHOTO" ] || { echo "Error: photo not found: $PHOTO" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
TS=$(date +%Y%m%d-%H%M%S)
WORK=$(mktemp -d -t "talk-as-me-sad-XXXXXX")
trap 'rm -rf "$WORK"' EXIT

echo "[1/2] Generating speech via Pocket TTS..."
if ! curl -fsS "${TTS_SERVER}/health" >/dev/null 2>&1; then
    echo "  Starting Pocket TTS server..."
    # shellcheck disable=SC1091
    source "$CHATTERBOX_VENV/bin/activate"
    nohup pocket-tts serve --voice "$VOICE_SAMPLE" >/tmp/pocket-tts-server.log 2>&1 &
    for _ in $(seq 1 30); do
        sleep 1
        curl -fsS "${TTS_SERVER}/health" >/dev/null 2>&1 && break
    done
fi

curl -fsS -X POST "${TTS_SERVER}/tts" \
    --form-string "text=${TEXT}" \
    --output "$WORK/speech.wav"

[ -s "$WORK/speech.wav" ] || { echo "Error: TTS produced empty audio" >&2; exit 1; }

# Apply locked narration EQ
ffmpeg -y -hide_banner -loglevel error \
    -i "$WORK/speech.wav" \
    -af "volume=1.35,equalizer=f=4000:width_type=h:width=2000:g=3,loudnorm=I=-16:LRA=11:TP=-1.5" \
    "$WORK/speech_eq.wav"

echo "[2/2] SadTalker --still on portrait (~1 min animation + 2-3 min GFPGAN enhance)..."
cd "$SAD_DIR"
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python inference.py \
    --driven_audio "$WORK/speech_eq.wav" \
    --source_image "$PHOTO" \
    --result_dir "$WORK/sad" \
    --still --preprocess full --enhancer gfpgan \
    >"$WORK/sad.log" 2>&1 || {
        echo "Error: SadTalker failed. Last lines:" >&2
        tail -20 "$WORK/sad.log" >&2
        exit 1
    }

# SadTalker writes the final mp4 at <result_dir>/<timestamp>.mp4
FINAL_SRC=$(find "$WORK/sad" -maxdepth 2 -name "*.mp4" -not -path "*/avatar*" -not -name "*##*" | head -1)
if [ -z "$FINAL_SRC" ]; then
    echo "Error: SadTalker did not produce final mp4" >&2
    find "$WORK/sad" -name "*.mp4" >&2
    exit 1
fi

FINAL="$OUTPUT_DIR/$TS.mp4"
cp "$FINAL_SRC" "$FINAL"

echo ""
echo "✓ Done: $FINAL"
