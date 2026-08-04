#!/bin/bash
# narrate-as-me — turn a text script into a talking-head video of Matt.
#
# Pipeline:
#   text -> Pocket TTS (Matt's cloned voice) -> speech.wav
#   photo + driving video -> LivePortrait -> silent_animation.mp4
#   silent + speech -> ffmpeg mux -> output/<timestamp>.mp4
#
# Usage:
#   narrate-as-me.sh "your script here" [photo.jpg] [driving.mp4]
#   narrate-as-me.sh --script-file path.txt [photo.jpg] [driving.mp4]
#
# Defaults if photo/driving not given:
#   photo:   ~/Desktop/Local AI Setup/scripts/avatar-source.jpg
#   driving: bundled LivePortrait example d0.mp4
#
# Limitation:
#   Lip sync follows the driving video's mouth, NOT the words being spoken.
#   For proper word-accurate lip sync, layer Wav2Lip in a follow-up pass.

set -euo pipefail

LP_DIR="$HOME/Desktop/Local AI Setup/LivePortrait"
SCRIPTS_DIR="$HOME/Desktop/Local AI Setup/scripts"
OUTPUT_DIR="$HOME/Desktop/Local AI Setup/output"
TTS_SERVER="http://localhost:8000"
VOICE_SAMPLE="$HOME/Library/Application Support/sh.voicebox.app/voice-sample-backup.wav"
CHATTERBOX_VENV="$HOME/chatterbox-env"
DEFAULT_PHOTO="$SCRIPTS_DIR/avatar-source.jpg"
DEFAULT_DRIVING="$LP_DIR/assets/examples/driving/d0.mp4"

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
Usage: $(basename "$0") "script text" [photo.jpg] [driving.mp4]
       $(basename "$0") --script-file path.txt [photo.jpg] [driving.mp4]

Drop a front-facing photo of yourself at:
  $DEFAULT_PHOTO
EOF
    exit 1
fi

PHOTO="${1:-$DEFAULT_PHOTO}"
DRIVING="${2:-$DEFAULT_DRIVING}"

[ -f "$PHOTO" ]   || { echo "Error: photo not found: $PHOTO" >&2; exit 1; }
[ -f "$DRIVING" ] || { echo "Error: driving video not found: $DRIVING" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
TS=$(date +%Y%m%d-%H%M%S)
WORK=$(mktemp -d -t "narrate-as-me-XXXXXX")
trap 'rm -rf "$WORK"' EXIT

echo "[1/3] Generating speech via Pocket TTS..."
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

# Apply Matt's locked narration EQ (volume +35%, 4 kHz +3 dB shelf, -16 LUFS)
ffmpeg -y -hide_banner -loglevel error \
    -i "$WORK/speech.wav" \
    -af "volume=1.35,equalizer=f=4000:width_type=h:width=2000:g=3,loudnorm=I=-16:LRA=11:TP=-1.5" \
    "$WORK/speech_eq.wav"

echo "[2/3] Animating with LivePortrait (this takes ~15-60s)..."
cd "$LP_DIR"
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python inference.py \
    -s "$PHOTO" \
    -d "$DRIVING" \
    --output-dir "$WORK/animations" \
    >"$WORK/lp.log" 2>&1 || {
        echo "Error: LivePortrait failed. Last lines of log:" >&2
        tail -20 "$WORK/lp.log" >&2
        exit 1
    }

# LivePortrait names output as <source_basename>--<driving_basename>.mp4
SRC_BASE=$(basename "$PHOTO"); SRC_BASE="${SRC_BASE%.*}"
DRV_BASE=$(basename "$DRIVING"); DRV_BASE="${DRV_BASE%.*}"
SILENT="$WORK/animations/${SRC_BASE}--${DRV_BASE}.mp4"

if [ ! -f "$SILENT" ]; then
    echo "Error: expected silent video not found: $SILENT" >&2
    echo "Files produced:" >&2
    ls -1 "$WORK/animations/" >&2
    exit 1
fi

echo "[3/3] Muxing audio onto video..."
FINAL="$OUTPUT_DIR/$TS.mp4"

# Loop the silent video to cover the full audio length, then trim to audio end
ffmpeg -y -hide_banner -loglevel error \
    -stream_loop -1 -i "$SILENT" \
    -i "$WORK/speech_eq.wav" \
    -map 0:v:0 -map 1:a:0 \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p \
    -c:a aac -b:a 192k \
    -shortest \
    "$FINAL"

echo ""
echo "✓ Done: $FINAL"
echo ""
echo "Note: lip movement matches the driving video, not the spoken words."
echo "      Layer Wav2Lip in a follow-up pass for word-accurate lip sync."
