"""
downloader.py — Download YouTube video (mp4) and extract audio (wav) via yt-dlp + ffmpeg.

Uses the ffmpeg binary bundled with imageio-ffmpeg so no system-level ffmpeg
installation is required on Windows/macOS/Linux.
"""
import os
import logging
import subprocess
import tempfile

import imageio_ffmpeg
import yt_dlp

logger = logging.getLogger(__name__)


def _ffmpeg_exe() -> str:
    """
    Return path to a working ffmpeg binary.

    yt-dlp looks for a file named exactly 'ffmpeg' / 'ffmpeg.exe' inside the
    directory pointed to by ffmpeg_location.  The imageio-ffmpeg bundle uses a
    versioned name (e.g. 'ffmpeg-win-x86_64-v7.1.exe'), so we copy it once to
    a temp dir under the standard name so both yt-dlp and subprocess can use it.
    """
    import shutil
    try:
        src = imageio_ffmpeg.get_ffmpeg_exe()
        if src and os.path.isfile(src):
            ffmpeg_dir = os.path.join(tempfile.gettempdir(), "ai_video_ffmpeg")
            os.makedirs(ffmpeg_dir, exist_ok=True)
            exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            dst = os.path.join(ffmpeg_dir, exe_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            return dst
    except Exception:
        pass
    return "ffmpeg"  # hope it's on PATH


def get_download_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), "ai_video_summarizer")
    os.makedirs(path, exist_ok=True)
    return path


def download_video(url: str) -> dict:
    """
    Download YouTube video as mp4 and extract 16 kHz mono WAV for Whisper.

    Returns:
        {video_path, audio_path, title, duration}
    """
    dl_dir = get_download_dir()
    video_path = os.path.join(dl_dir, "video.mp4")
    audio_path = os.path.join(dl_dir, "audio.wav")
    ffmpeg = _ffmpeg_exe()

    logger.info("Using ffmpeg: %s", ffmpeg)

    # Remove stale files from a prior run
    for p in [video_path, audio_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    logger.info("Downloading: %s", url)

    ydl_opts = {
        # Best mp4 with merged audio — requires ffmpeg for muxing
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": video_path,
        "merge_output_format": "mp4",
        "ffmpeg_location": os.path.dirname(ffmpeg),  # tell yt-dlp where ffmpeg lives
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title    = info.get("title", "Unknown")
        duration = int(info.get("duration") or 0)

    # yt-dlp may append a format suffix; find whatever was written
    if not os.path.exists(video_path):
        candidates = [
            f for f in os.listdir(dl_dir)
            if f.startswith("video") and f.endswith((".mp4", ".mkv", ".webm"))
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Downloaded video not found in {dl_dir}. "
                "Check that the URL is valid and yt-dlp completed without error."
            )
        os.rename(os.path.join(dl_dir, candidates[0]), video_path)

    logger.info("Downloaded: '%s' (%d s)", title, duration)

    # Extract 16 kHz mono WAV — Whisper's preferred input format
    logger.info("Extracting audio to WAV...")
    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed:\n{proc.stderr[-600:]}"
        )

    logger.info("Audio ready: %s", audio_path)
    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "title":      title,
        "duration":   duration,
    }
