Transcribe Meeting Recording (Offline)

Quick start

1) Prereqs
- macOS with Homebrew
- Python 3.10+

2) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
brew install ffmpeg
```

3) Transcribe (your file)

```bash
python transcribe.py --input /path/to/video.mp4 --output-dir outputs --model large-v3
```

Artifacts
- TXT/SRT/VTT/JSON under `outputs/<basename>.*`
- QC report at `outputs/<basename>_qc.txt`
- Live progress at `outputs/<basename>_progress.json`

Faster pass

```bash
python transcribe.py --input /path/to/video.mp4 --output-dir outputs_fast --model medium --beam-size 1 --no-vad --no-word-timestamps
```

Notes
- First run downloads the model (one-time). Fully offline after.
- Uses faster-whisper with local ONNX/ctranslate2; no cloud calls.

Demo

```bash
# fetch a small demo video
./scripts/fetch_demo.sh

# run on demo
python transcribe.py --input assets/demo.mp4 --output-dir outputs_demo --model tiny --beam-size 1 --no-word-timestamps
```



