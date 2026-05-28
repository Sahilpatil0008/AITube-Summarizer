"""
pipeline.py — Full AI video summarizer pipeline.

Usage:
    python pipeline.py <youtube_url> [--output output/summary.mp4]

Steps:
    1. Download video + extract audio  (downloader.py)
    2. Transcribe audio with Whisper   (transcriber.py)
    3. BART summarization + SBERT rank (summarizer.py)
    4. Multi-signal scoring            (scorer.py)
    5. Select clips & merge video      (video_cutter.py)
"""
import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI YouTube Video Summarizer")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output", default="output/summary.mp4",
        help="Output video path (default: output/summary.mp4)"
    )
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    logger.info("=== Step 1/4: Downloading video ===")
    from downloader import download_video
    media = download_video(args.url)
    logger.info("Title: %s  (%d s)", media["title"], media["duration"])

    # ── Step 2: Transcribe ────────────────────────────────────────────────────
    logger.info("=== Step 2/4: Transcribing with Whisper ===")
    from transcriber import transcribe_audio
    segments = transcribe_audio(media["audio_path"])
    logger.info("Transcribed %d segments.", len(segments))

    # ── Step 3: BART summarization + semantic ranking ─────────────────────────
    logger.info("=== Step 3/4: Summarizing & ranking ===")
    from summarizer import summarize_transcript
    result = summarize_transcript(segments)
    logger.info("Summary (%d chars):\n%s", len(result["summary"]), result["summary"])

    # ── Step 4a: Multi-signal scoring ─────────────────────────────────────────
    logger.info("=== Step 4/4: Scoring & cutting ===")
    from scorer import score_segments
    scored = score_segments(
        result["ranked_segments"],
        audio_path=media["audio_path"],
        video_path=media["video_path"],
    )

    # ── Step 4b: Select clips & assemble final video ──────────────────────────
    from video_cutter import select_segments, cut_and_merge
    selected, total_dur = select_segments(scored)
    out_path = cut_and_merge(media["video_path"], selected, args.output)

    logger.info("")
    logger.info("=== Done! ===")
    logger.info("Output : %s", out_path)
    logger.info("Length : %.1f seconds", total_dur)
    logger.info("Summary: %s", result["summary"])


if __name__ == "__main__":
    main()
