"""
scorer.py — Multi-signal segment scorer.

Four signals are combined with a weighted average:
    NLP importance  (TF-IDF mean)          — weight 0.35
    Semantic score  (BART/SBERT cosine)    — weight 0.30
    Audio energy    (librosa RMS)          — weight 0.20
    Visual change   (OpenCV frame diff)    — weight 0.15
"""
import logging
from typing import Any, Dict, List

import cv2
import librosa
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

WEIGHTS = {"nlp": 0.35, "semantic": 0.30, "audio": 0.20, "visual": 0.15}
# Cap frames sampled per segment to keep visual analysis fast on long videos
MAX_FRAMES_PER_SEGMENT = 60


# ── normalisation ─────────────────────────────────────────────────────────────

def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


# ── individual signal computers ───────────────────────────────────────────────

def _tfidf_scores(segments: List[Dict[str, Any]]) -> np.ndarray:
    """Mean TF-IDF weight per segment — captures keyword density."""
    texts = [s["text"] for s in segments]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform(texts)
    return np.asarray(mat.mean(axis=1)).flatten()


def _audio_energy_scores(
    audio_path: str, segments: List[Dict[str, Any]]
) -> np.ndarray:
    """Root-mean-square energy of the waveform slice for each segment."""
    logger.info("Analysing audio energy (%d segments)...", len(segments))
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    scores: List[float] = []
    for seg in segments:
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        clip = y[s:e]
        rms = float(np.sqrt(np.mean(clip ** 2))) if len(clip) > 0 else 0.0
        scores.append(rms)
    return np.array(scores, dtype=float)


def _visual_change_scores(
    video_path: str, segments: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Mean absolute pixel difference between consecutive (sampled) frames.
    Higher value → more on-screen motion → more visually dynamic.
    """
    logger.info("Analysing visual change (%d segments)...", len(segments))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores: List[float] = []

    for seg in segments:
        start_f = int(seg["start"] * fps)
        end_f = min(int(seg["end"] * fps), total_frames - 1)
        span = end_f - start_f

        if span <= 0:
            scores.append(0.0)
            continue

        # Sample evenly; never read more than MAX_FRAMES_PER_SEGMENT frames
        step = max(1, span // MAX_FRAMES_PER_SEGMENT)
        frame_indices = range(start_f, end_f, step)

        prev_gray: np.ndarray | None = None
        diffs: List[float] = []

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                break
            # Downsample before diff to keep memory and CPU usage low
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diffs.append(
                    float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
                )
            prev_gray = gray

        scores.append(float(np.mean(diffs)) if diffs else 0.0)

    cap.release()
    return np.array(scores, dtype=float)


# ── public API ────────────────────────────────────────────────────────────────

def score_segments(
    segments: List[Dict[str, Any]],
    audio_path: str,
    video_path: str,
) -> List[Dict[str, Any]]:
    """
    Score every segment and return them sorted by *combined_score* (descending).

    Args:
        segments:    List from summarizer.py — already contain *semantic_score*.
        audio_path:  Path to the WAV file.
        video_path:  Path to the mp4 file.

    Returns:
        Same list enriched with nlp_score, audio_score, visual_score,
        semantic_score (normalised), and combined_score; sorted desc.
    """
    n = len(segments)
    logger.info("Scoring %d segments with 4 signals...", n)

    nlp_raw = _tfidf_scores(segments)
    sem_raw = np.array([s.get("semantic_score", 0.0) for s in segments], dtype=float)
    aud_raw = _audio_energy_scores(audio_path, segments)
    vis_raw = _visual_change_scores(video_path, segments)

    nlp  = _minmax(nlp_raw)
    sem  = _minmax(sem_raw)
    aud  = _minmax(aud_raw)
    vis  = _minmax(vis_raw)

    combined = (
        WEIGHTS["nlp"]      * nlp
        + WEIGHTS["semantic"] * sem
        + WEIGHTS["audio"]    * aud
        + WEIGHTS["visual"]   * vis
    )

    scored: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        scored.append(
            {
                **seg,
                "nlp_score":      float(nlp[i]),
                "semantic_score": float(sem[i]),
                "audio_score":    float(aud[i]),
                "visual_score":   float(vis[i]),
                "combined_score": float(combined[i]),
            }
        )

    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    logger.info(
        "Top segment — combined=%.4f | '%s'",
        scored[0]["combined_score"],
        scored[0]["text"][:80],
    )
    return scored
