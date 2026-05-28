"""
video_cutter.py — Select the best-scored segments, clip them from the source
video and concatenate with smooth crossfade transitions using MoviePy 1.x.

Target output duration: 60 – 120 seconds.
"""
import logging
import os
from typing import Any, Dict, List, Tuple

from moviepy.editor import VideoFileClip, concatenate_videoclips

logger = logging.getLogger(__name__)

MIN_DURATION  = 60.0
MAX_DURATION  = 120.0
CROSSFADE_DUR = 0.3
FADEIN_DUR    = 0.5
FADEOUT_DUR   = 0.5
MIN_SEG_DUR   = 1.5   # skip segments shorter than this


# ── segment selection ─────────────────────────────────────────────────────────

def select_segments(
    scored_segments: List[Dict[str, Any]],
    min_dur: float = MIN_DURATION,
    max_dur: float = MAX_DURATION,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Greedily pick the highest-scoring, non-overlapping segments until the
    accumulated wall-clock duration lands in [min_dur, max_dur].

    Returns:
        (segments_sorted_by_timestamp, total_selected_duration)
    """
    if not scored_segments:
        raise ValueError("scored_segments is empty — nothing to select.")

    selected: List[Dict[str, Any]] = []
    accepted_intervals: List[Tuple[float, float]] = []
    total = 0.0

    for seg in scored_segments:   # already sorted by combined_score desc
        dur = seg["end"] - seg["start"]
        if dur < MIN_SEG_DUR:
            continue
        if not seg["text"].strip():
            continue

        # Skip if this segment overlaps an already-accepted one
        if any(seg["start"] < e and seg["end"] > s for s, e in accepted_intervals):
            continue

        budget = max_dur - total
        if dur > budget:
            if total < min_dur:
                # Trim tail to fill remaining budget exactly
                seg = {**seg, "end": seg["start"] + budget}
                dur = budget
            else:
                continue

        selected.append(seg)
        accepted_intervals.append((seg["start"], seg["end"]))
        total += dur

        if total >= max_dur:
            break

    if not selected:
        raise ValueError("No valid segments could be selected (all too short or empty).")

    selected.sort(key=lambda x: x["start"])
    logger.info(
        "Selected %d segment(s) | %.1f s total (target %.0f–%.0f s)",
        len(selected), total, min_dur, max_dur,
    )
    return selected, total


# ── video assembly ────────────────────────────────────────────────────────────

def cut_and_merge(
    video_path: str,
    selected_segments: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """
    Extract *selected_segments* from *video_path*, apply crossfade transitions
    and a global fade-in / fade-out, then write to *output_path*.

    Returns:
        Absolute path of the saved output file.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Opening source video: %s", video_path)
    source = VideoFileClip(video_path)

    raw_clips = []
    for seg in selected_segments:
        start = max(0.0, float(seg["start"]))
        end   = min(source.duration, float(seg["end"]))
        if end - start < 0.5:
            logger.warning("Skipping near-zero-length segment [%.2f, %.2f]", start, end)
            continue
        raw_clips.append(source.subclip(start, end))

    if not raw_clips:
        source.close()
        raise ValueError("All extracted clips were zero-length. Check segment timestamps.")

    logger.info("Applying fade / crossfade effects to %d clip(s)...", len(raw_clips))

    if len(raw_clips) == 1:
        final = raw_clips[0].fadein(FADEIN_DUR).fadeout(FADEOUT_DUR)
    else:
        processed = []
        for i, clip in enumerate(raw_clips):
            if i == 0:
                c = clip.fadein(FADEIN_DUR).crossfadeout(CROSSFADE_DUR)
            elif i == len(raw_clips) - 1:
                c = clip.crossfadein(CROSSFADE_DUR).fadeout(FADEOUT_DUR)
            else:
                c = clip.crossfadein(CROSSFADE_DUR).crossfadeout(CROSSFADE_DUR)
            processed.append(c)

        # padding=-CROSSFADE_DUR overlaps consecutive clips so the crossfades blend
        final = concatenate_videoclips(processed, padding=-CROSSFADE_DUR, method="compose")

    logger.info("Writing output video: %s", output_path)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        remove_temp=True,
        logger=None,
    )

    out_duration = final.duration

    final.close()
    for c in raw_clips:
        c.close()
    source.close()

    logger.info("Saved: %s (%.1f seconds)", output_path, out_duration)
    return output_path
