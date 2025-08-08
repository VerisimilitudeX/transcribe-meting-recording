#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="assets"
mkdir -p "$DEST_DIR"
AUDIO_AIFF="$DEST_DIR/speech.aiff"
VIDEO_MP4="$DEST_DIR/speech.mp4"

TEXT="This is a demo meeting recording with spoken words generated locally. It demonstrates the offline transcription pipeline with timestamps, captions, and a quality report."

if ! command -v say >/dev/null 2>&1; then
  echo "macOS 'say' command not found." >&2
  exit 1
fi

echo "Generating speech AIFF via macOS TTS..."
say -v Alex -o "$AUDIO_AIFF" "$TEXT"

echo "Wrapping AIFF into MP4 with a black background..."
ffmpeg -y -f lavfi -i color=c=black:s=640x360:r=25 -i "$AUDIO_AIFF" -shortest \
  -c:v libx264 -tune stillimage -pix_fmt yuv420p -c:a aac -movflags +faststart "$VIDEO_MP4" </dev/null

echo "Demo video created at: $VIDEO_MP4"

