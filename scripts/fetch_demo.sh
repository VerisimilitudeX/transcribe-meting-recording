#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="assets"
DEST_FILE="$DEST_DIR/demo.mp4"
URL="https://sample-videos.com/video321/mp4/240/big_buck_bunny_240p_1mb.mp4"

mkdir -p "$DEST_DIR"
echo "Downloading demo video to $DEST_FILE ..."
curl -L "$URL" -o "$DEST_FILE"
echo "Done."

