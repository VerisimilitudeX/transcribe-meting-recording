#!/usr/bin/env python3
import argparse
import json
import os
import sys
import math
import subprocess
from datetime import timedelta
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from faster_whisper import WhisperModel
import srt as srt_lib
import webvtt


console = Console()


def format_timestamp_srt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def get_media_duration_seconds(input_path: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                input_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def save_txt(segments: List[Dict[str, Any]], out_txt: str) -> None:
    with open(out_txt, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg["text"].strip() + "\n")


def save_srt(segments: List[Dict[str, Any]], out_srt: str) -> None:
    items = []
    for i, seg in enumerate(segments, start=1):
        start = timedelta(seconds=float(seg["start"]))
        end = timedelta(seconds=float(seg["end"]))
        items.append(
            srt_lib.Subtitle(index=i, start=start, end=end, content=seg["text"].strip())
        )
    with open(out_srt, "w", encoding="utf-8") as f:
        f.write(srt_lib.compose(items))


def save_vtt(segments: List[Dict[str, Any]], out_vtt: str) -> None:
    vtt = webvtt.WebVTT()
    for seg in segments:
        start = format_timestamp_vtt(seg["start"])
        end = format_timestamp_vtt(seg["end"])
        vtt.captions.append(webvtt.Caption(start, end, seg["text"].strip()))
    vtt.save(out_vtt)


def save_json(segments: List[Dict[str, Any]], meta: Dict[str, Any], out_json: str) -> None:
    payload = {
        "metadata": meta,
        "segments": segments,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_qc_report(segments: List[Dict[str, Any]], meta: Dict[str, Any], out_report: str) -> None:
    overall_confidences = [seg.get("avg_prob", 0.0) for seg in segments if seg.get("avg_prob") is not None]
    overall_avg = sum(overall_confidences) / len(overall_confidences) if overall_confidences else 0.0

    # Flag segments with low average probability
    threshold = 0.60
    low_conf = [s for s in segments if s.get("avg_prob", 1.0) < threshold]
    low_conf.sort(key=lambda s: s.get("avg_prob", 1.0))

    total_speech_seconds = sum(max(0.0, float(s["end"]) - float(s["start"])) for s in segments)
    media_duration = meta.get("media_duration_seconds") or 0.0
    speech_coverage = (total_speech_seconds / media_duration) if media_duration > 0 else 0.0

    lines = []
    lines.append("QC Report")
    lines.append("=========")
    lines.append("")
    lines.append(f"Language: {meta.get('language')}  (prob={meta.get('language_probability')})")
    lines.append(f"Media duration: {media_duration:.2f}s")
    lines.append(f"Segments: {len(segments)}")
    lines.append(f"Overall average confidence: {overall_avg:.3f}")
    lines.append(f"Estimated speech coverage: {speech_coverage*100:.1f}% of media")
    lines.append("")
    lines.append(f"Low-confidence segments (avg_prob < {threshold:.2f}): {len(low_conf)}")
    lines.append("")
    for seg in low_conf[:100]:  # cap for readability
        start = format_timestamp_srt(seg["start"])  # SRT-like
        end = format_timestamp_srt(seg["end"])      # SRT-like
        lines.append(f"[{start} --> {end}] conf={seg.get('avg_prob', 0.0):.3f}")
        snippet = seg["text"].strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        lines.append(f"  {snippet}")
        lines.append("")

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def transcribe(
    input_path: str,
    output_dir: str,
    model_size: str = "large-v3",
    language: str | None = None,
    compute_type: str = "auto",
    beam_size: int = 5,
    vad: bool = True,
    word_timestamps: bool = True,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    # Precompute media info and output base paths so we can emit progress early
    media_duration = get_media_duration_seconds(input_path)
    base = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0])
    out_txt = base + ".txt"
    out_srt = base + ".srt"
    out_vtt = base + ".vtt"
    out_json = base + ".json"
    out_report = base + "_qc.txt"
    out_progress = base + "_progress.json"

    # Initialize progress file
    try:
        with open(out_progress, "w", encoding="utf-8") as pf:
            json.dump({
                "status": "starting",
                "processed_seconds": 0.0,
                "percent": 0.0,
                "media_duration_seconds": media_duration,
            }, pf)
    except Exception:
        pass

    console.log(f"Loading model: {model_size}")
    model = WhisperModel(
        model_size_or_path=model_size,
        device="auto",
        compute_type=compute_type,
        cpu_threads=max(1, os.cpu_count() or 1),
        num_workers=1,
    )

    vad_params = {"min_silence_duration_ms": 500}

    console.log("Starting transcription...")
    with Progress() as progress:
        task = progress.add_task("Transcribing", total=None)
        segments_iter, info = model.transcribe(
            input_path,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=None,
            patience=1.0,
            length_penalty=1.0,
            repetition_penalty=1.0,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            word_timestamps=word_timestamps,
            vad_filter=vad,
            vad_parameters=vad_params,
        )

        collected: List[Dict[str, Any]] = []
        processed_count = 0
        for seg in segments_iter:
            # Compute average probability from words if available, else from segment avg_logprob
            words = []
            avg_prob = None
            if getattr(seg, "words", None):
                prob_values = []
                for w in seg.words:
                    w_prob = None
                    # faster-whisper exposes prob as w.probability (0..1) when available
                    if hasattr(w, "probability") and w.probability is not None:
                        w_prob = float(w.probability)
                    words.append(
                        {
                            "start": float(w.start) if w.start is not None else None,
                            "end": float(w.end) if w.end is not None else None,
                            "word": w.word,
                            "prob": w_prob,
                        }
                    )
                    if w_prob is not None:
                        prob_values.append(float(w_prob))
                if prob_values:
                    avg_prob = sum(prob_values) / len(prob_values)
            if avg_prob is None and seg.avg_logprob is not None:
                # Convert average logprob roughly to a probability-like value
                avg_prob = 1.0 / (1.0 + math.exp(-float(seg.avg_logprob)))

            collected.append(
                {
                    "id": seg.id,
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text,
                    "avg_prob": avg_prob,
                    "words": words,
                }
            )
            processed_count += 1
            # Progress logging each segment
            current_sec = float(seg.end)
            percent = (current_sec / media_duration * 100.0) if media_duration > 0 else None
            console.log(
                f"Processed {processed_count} segments, current time {format_timestamp_srt(seg.end)}"
                + (f" ({percent:.1f}% of media)" if percent is not None else "")
            )
            try:
                with open(out_progress, "w", encoding="utf-8") as pf:
                    json.dump({
                        "status": "running",
                        "processed_seconds": current_sec,
                        "percent": percent,
                        "media_duration_seconds": media_duration,
                        "last_update_segment": processed_count,
                    }, pf)
            except Exception:
                pass
            progress.advance(task)

    meta = {
        "model": model_size,
        "language": info.language,
        "language_probability": getattr(info, "language_probability", None),
        "media_duration_seconds": media_duration,
        "beam_size": beam_size,
        "vad_filter": vad,
        "word_timestamps": word_timestamps,
        "compute_type": compute_type,
        "source": os.path.abspath(input_path),
    }

    # Outputs
    console.log("Writing outputs (txt, srt, vtt, json, qc)...")
    save_txt(collected, out_txt)
    save_srt(collected, out_srt)
    save_vtt(collected, out_vtt)
    save_json(collected, meta, out_json)
    save_qc_report(collected, meta, out_report)

    # Finalize progress
    try:
        with open(out_progress, "w", encoding="utf-8") as pf:
            json.dump({
                "status": "done",
                "processed_seconds": media_duration,
                "percent": 100.0 if media_duration > 0 else None,
                "media_duration_seconds": media_duration,
                "last_update_segment": processed_count,
            }, pf)
    except Exception:
        pass

    # Short console summary
    table = Table(title="Transcription Summary")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("Language", str(meta.get("language")))
    table.add_row("Media duration (s)", f"{media_duration:.2f}")
    confidences = [s.get("avg_prob", 0.0) for s in collected if s.get("avg_prob") is not None]
    overall_avg = sum(confidences) / len(confidences) if confidences else 0.0
    table.add_row("Overall avg conf", f"{overall_avg:.3f}")
    table.add_row("Segments", str(len(collected)))
    table.add_row("TXT", out_txt)
    table.add_row("SRT", out_srt)
    table.add_row("VTT", out_vtt)
    table.add_row("JSON", out_json)
    table.add_row("QC Report", out_report)
    console.print(table)

    return {
        "segments": collected,
        "meta": meta,
        "paths": {
            "txt": out_txt,
            "srt": out_srt,
            "vtt": out_vtt,
            "json": out_json,
            "qc": out_report,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Offline transcription with QC report")
    parser.add_argument("--input", required=True, help="Path to input audio/video file")
    parser.add_argument(
        "--output-dir", default="outputs", help="Directory to write outputs"
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size or local CTranslate2 path (e.g., tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--language", default=None, help="Language code (auto-detect if omitted)"
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type (auto, float16, int8, int8_float16, etc.)",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    parser.add_argument(
        "--no-word-timestamps",
        action="store_true",
        help="Disable word-level timestamps to speed up",
    )

    args = parser.parse_args()
    if not os.path.isfile(args.input):
        console.print(f"[red]Input not found:[/red] {args.input}")
        sys.exit(1)

    try:
        transcribe(
            input_path=args.input,
            output_dir=args.output_dir,
            model_size=args.model,
            language=args.language,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            vad=not args.no_vad,
            word_timestamps=not args.no_word_timestamps,
        )
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()


