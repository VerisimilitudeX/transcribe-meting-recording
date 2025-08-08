#!/usr/bin/env bash
set -euo pipefail
DEST_DIR="assets"
AUDIO="$DEST_DIR/speech.wav"
VIDEO="$DEST_DIR/speech.mp4"
URL="https://file-examples.com/storage/fef57276e83b2ad846d10c5/2017/11/file_example_WAV_1MG.wav"
mkdir -p "$DEST_DIR"
echo "Downloading speech sample..."
curl -L "$URL" -o "$AUDIO"
echo "Wrapping into MP4 (with static image) ..."
ffmpeg -y -f lavfi -i color=c=black:s=640x360:d=60 -i "$AUDIO" -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac "$VIDEO" </dev/null
echo "Saved demo: $VIDEO"
