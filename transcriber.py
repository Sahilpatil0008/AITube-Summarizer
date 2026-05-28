"""
transcriber.py — Transcribe audio to timestamped segments via OpenAI Whisper.
"""
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List

import whisper


def _ensure_ffmpeg_on_path() -> None:
    """
    Whisper calls ffmpeg via subprocess using the bare name 'ffmpeg'.
    If the system doesn't have it on PATH, inject the imageio-ffmpeg
    bundled binary by placing it (or a copy named 'ffmpeg.exe') on PATH.
    """
    if shutil.which("ffmpeg"):
        return  # already available
    try:
        import imageio_ffmpeg
        src = imageio_ffmpeg.get_ffmpeg_exe()
        if not src or not os.path.isfile(src):
            return
        ffmpeg_dir = os.path.join(tempfile.gettempdir(), "ai_video_ffmpeg")
        os.makedirs(ffmpeg_dir, exist_ok=True)
        exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        dst = os.path.join(ffmpeg_dir, exe_name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        # Prepend to PATH so Whisper's subprocess.run(['ffmpeg', ...]) finds it
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        logging.getLogger(__name__).info("Added bundled ffmpeg to PATH: %s", ffmpeg_dir)
    except Exception as exc:
        logging.getLogger(__name__).warning("Could not inject ffmpeg into PATH: %s", exc)

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
) -> List[Dict[str, Any]]:
    """
    Transcribe *audio_path* with Whisper and return timestamped segments.

    Args:
        audio_path:  Path to a WAV file (16 kHz mono works best).
        model_size:  Whisper model variant — tiny | base | small | medium.

    Returns:
        List of dicts: [{start: float, end: float, text: str}, ...]
    """
    _ensure_ffmpeg_on_path()

    logger.info("Loading Whisper model '%s'...", model_size)
    model = whisper.load_model(model_size)

    logger.info("Transcribing: %s", audio_path)
    result = model.transcribe(
        audio_path,
        verbose=False,
        fp16=False,      # CPU-safe; set True only when a CUDA GPU is available
        language=None,   # auto-detect language
    )

    segments: List[Dict[str, Any]] = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": text,
            }
        )

    if not segments:
        raise ValueError(
            "Whisper returned no segments. "
            "The audio may be silent or in an unsupported language."
        )

    lang = result.get("language", "unknown")
    logger.info(
        "Transcription complete — %d segments, detected language: %s",
        len(segments),
        lang,
    )
    return segments
