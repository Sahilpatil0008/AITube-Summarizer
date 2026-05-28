"""
AI Video Summarizer — two-phase pipeline.
Phase 1 (Analyze):  transcript + metadata + scoring + summary  — no download needed
Phase 2 (Build):    download only the selected clips via yt-dlp download_ranges
"""

# ── SSL patch — must be FIRST, before any network imports ─────────────────────
# HF Spaces / Docker environments suffer SSL: UNEXPECTED_EOF_WHILE_READING
# because of network proxying. Patch ssl, requests, and urllib3 globally.
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("PYTHONHTTPSVERIFY",  "0")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("CURL_CA_BUNDLE",     "")

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

try:
    import requests
    from requests.adapters import HTTPAdapter
    _s = requests.Session()
    _s.verify = False
    # Monkey-patch so all requests.get / requests.post skip verification
    _orig_request = requests.Session.request
    def _patched_request(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        return _orig_request(self, *args, **kwargs)
    requests.Session.request = _patched_request
except Exception:
    pass
# ──────────────────────────────────────────────────────────────────────────────

import re, json, uuid, time, queue, shutil, tempfile, threading, subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

import numpy as np
from flask import Flask, request, jsonify, Response, render_template, send_file

os.chdir(os.path.dirname(os.path.abspath(__file__)))
app  = Flask(__name__)
jobs: dict = {}

# ── YouTube cookie support (required on cloud/HF Spaces — YouTube blocks IPs) ─
# Set the HF Space secret  YOUTUBE_COOKIES  to the content of a Netscape
# cookies.txt exported from a logged-in Chrome/Firefox session.
_COOKIE_FILE = "/tmp/yt_cookies.txt"

def _setup_cookies() -> bool:
    """Write YOUTUBE_COOKIES env-secret to a temp file. Returns True if available."""
    content = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if content:
        with open(_COOKIE_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False

_HAS_COOKIES = _setup_cookies()

# ── Common yt-dlp options ──────────────────────────────────────────────────────
_YDL_BASE: dict = {
    "quiet":              True,
    "no_warnings":        True,
    "nocheckcertificate": True,
    "socket_timeout":     30,
    "retries":            5,
    "fragment_retries":   5,
    # Use multiple clients: tv_embedded is less restricted on cloud IPs
    "extractor_args": {
        "youtube": {
            "player_client": ["tv_embedded", "android", "web"],
        }
    },
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 11; Pixel 5) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.91 Mobile Safari/537.36"
        )
    },
}

# Attach cookies if available (bypasses YouTube cloud-IP blocks)
if _HAS_COOKIES:
    _YDL_BASE["cookiefile"] = _COOKIE_FILE

# ── Global SBERT model (pre-warmed at startup for fast summaries) ─────────────
_sbert      = None
_sbert_lock = threading.Lock()

def _get_sbert():
    global _sbert
    with _sbert_lock:
        if _sbert is None:
            from sentence_transformers import SentenceTransformer
            _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert

threading.Thread(target=_get_sbert, daemon=True).start()   # pre-warm on startup


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _get_ffmpeg() -> str:
    import shutil as sh
    if sh.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        src = imageio_ffmpeg.get_ffmpeg_exe()
        if src and os.path.isfile(src):
            d   = os.path.join(tempfile.gettempdir(), "ai_video_ffmpeg")
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            return dst
    except Exception:
        pass
    return "ffmpeg"


def _get_video_id(url: str) -> Optional[str]:
    from urllib.parse import urlparse, parse_qs
    url = url.strip()
    if "/shorts/" in url:                                   # youtube.com/shorts/ID
        return url.split("/shorts/")[-1].split("?")[0].split("&")[0]
    if "youtu.be" in url:                                   # youtu.be/ID
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    return parse_qs(urlparse(url).query).get("v", [None])[0]


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a, dtype=float)


def _smooth(a: np.ndarray, w: int = 5) -> np.ndarray:
    if len(a) < w:
        return a.astype(float)
    return np.convolve(a.astype(float), np.ones(w) / w, mode="same")


def _snap_to_beat(t: float, beats: np.ndarray, tol: float = 1.5) -> float:
    if beats.size == 0:
        return t
    idx = int(np.argmin(np.abs(beats - t)))
    return float(beats[idx]) if abs(beats[idx] - t) <= tol else t


def _transcript_quality(segs: list) -> float:
    if not segs:
        return 0.0
    good = sum(1 for s in segs
               if len(s["text"].split()) >= 4
               and re.search(r"[a-zA-Z]{3,}", s["text"])
               and not re.match(r"^[\W\d♪♫\s]+$", s["text"]))
    return good / len(segs)


# ═══════════════════════════════════════════════════════════════════════════════
#  Audio analysis  (for music videos only — downloads audio stream, not full video)
# ═══════════════════════════════════════════════════════════════════════════════

def download_audio_for_analysis(url: str, out_path: str,
                                 ffmpeg_dir: str, max_sec: int = 300) -> Optional[str]:
    """Download bestaudio stream (small) and convert to 11025 Hz WAV."""
    import yt_dlp
    try:
        ydl_opts = {
            **_YDL_BASE,
            "format":          "bestaudio/best",
            "outtmpl":         out_path + ".%(ext)s",
            "ffmpeg_location": ffmpeg_dir,
            "download_ranges": yt_dlp.utils.download_range_func(None, [(0, max_sec)]),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # Find downloaded audio file
        folder = os.path.dirname(out_path)
        base   = os.path.basename(out_path)
        found  = [f for f in os.listdir(folder) if f.startswith(base) and not f.endswith(".wav")]
        if not found:
            return None
        src = os.path.join(folder, found[0])
        # Convert to WAV
        wav = out_path + ".wav"
        ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        r = subprocess.run([ffmpeg, "-y", "-i", src, "-vn",
                            "-acodec", "pcm_s16le", "-ar", "11025", "-ac", "1", wav],
                           capture_output=True, timeout=120)
        try: os.remove(src)
        except Exception: pass
        return wav if r.returncode == 0 and os.path.exists(wav) else None
    except Exception:
        return None


def analyze_audio(wav_path: str) -> dict:
    """Fast librosa analysis at sr=11025 with multi-level fallbacks."""
    import librosa
    try:
        y, sr = librosa.load(wav_path, sr=11025, mono=True)
    except Exception:
        return _empty_audio()

    hop   = 256
    fps_a = sr / hop
    dur   = len(y) / sr
    n     = int(dur) + 1

    try:
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    except Exception:
        rms = np.abs(y[::hop])

    try:
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    except Exception:
        onset = np.gradient(np.abs(rms).astype(float))

    beat_times, tempo = np.array([]), 120.0
    try:
        t_arr, bf = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="frames")
        tempo      = float(np.atleast_1d(t_arr)[0])
        beat_times = librosa.frames_to_time(bf, sr=sr, hop_length=hop)
    except Exception:
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(rms, distance=int(fps_a * 0.25),
                                  prominence=rms.std() * 0.5)
            beat_times = peaks / fps_a
            if len(beat_times) > 2:
                tempo = 60.0 / float(np.mean(np.diff(beat_times)))
        except Exception:
            pass

    def to_sec(arr):
        out = np.zeros(n)
        for i in range(n):
            s = int(i * fps_a); e = int((i + 1) * fps_a)
            if s < len(arr):
                out[i] = float(np.mean(arr[s:min(e, len(arr))]))
        return out

    return {"rms": to_sec(rms), "onset_env": to_sec(onset),
            "beat_times": beat_times, "tempo": tempo, "duration": dur}


def _empty_audio():
    return {"rms": np.array([]), "onset_env": np.array([]),
            "beat_times": np.array([]), "tempo": 0.0, "duration": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
#  Video-type detection
# ═══════════════════════════════════════════════════════════════════════════════

_MUSIC_KW = [
    "official video", "music video", "official music", " mv ", "( mv)",
    "lyrics", "lyric video", "ft.", "feat.", "remix", "single", "album",
    "cover", "live performance", "official audio", "visualizer", "vevo",
    "prod.", "produced by", "audio only", "full song",
]
_TUTORIAL_KW = [
    "tutorial", "how to", "how-to", "guide", "learn", "course", "lesson",
    "explained", "step by step", "beginner", "masterclass", "tips", "tricks",
    "review", "unboxing", "setup", "install",
]


def detect_video_type(title: str, description: str, tags: str,
                      duration: float, aspect_ratio: float,
                      tq: float, url: str = "") -> str:
    text = (title + " " + description[:500] + " " + tags).lower()
    is_short = (aspect_ratio < 1.0 or duration <= 180 or "/shorts/" in url
                or "#shorts" in text or "youtube shorts" in text)
    if is_short:
        return "short"
    if any(k in text for k in _MUSIC_KW) or tq < 0.20:
        return "music"
    if tq > 0.45 and any(k in text for k in _TUTORIAL_KW):
        return "speech"
    if tq < 0.35:
        return "mixed"
    return "speech"


# ═══════════════════════════════════════════════════════════════════════════════
#  Semantic summary  (sentence-transformers + KMeans clustering)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary(segs: list, title: str = "", n: int = 5) -> str:
    """
    High-quality extractive summary using semantic clustering:
    1. Pre-filter sentences by TF-IDF (keep top 120)
    2. Embed with sentence-transformers (all-MiniLM-L6-v2)
    3. KMeans into n topic clusters
    4. Pick the sentence closest to each cluster centroid
    5. Return chronologically ordered paragraph
    """
    good = [s for s in segs
            if len(s["text"].split()) >= 8
            and re.search(r"[a-zA-Z]{4,}", s["text"])
            and not re.match(r"^[\W\d♪♫\s]+$", s["text"])]
    if len(good) < 3:
        return " ".join(s["text"].strip() for s in good[:3])

    texts  = [s["text"].strip() for s in good]
    starts = [float(s.get("start", 0)) for s in good]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        # ── Pre-filter to top 50 by TF-IDF (fast on CPU) ─────────────────
        if len(texts) > 50:
            vec    = TfidfVectorizer(stop_words="english", max_features=4000)
            scores = np.asarray(vec.fit_transform(texts).mean(axis=1)).flatten()
            keep   = np.argsort(scores)[-50:]
            texts  = [texts[i] for i in keep]
            starts = [starts[i] for i in keep]

        # ── Semantic embeddings ────────────────────────────────────────────
        sbert = _get_sbert()
        if sbert is None:
            raise RuntimeError("SBERT not ready")
        embeddings = sbert.encode(texts, show_progress_bar=False,
                                  batch_size=64, normalize_embeddings=True)

        # ── KMeans topic clustering (n_init=3 for speed) ──────────────────
        k  = min(n, len(texts))
        km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
        km.fit(embeddings)

        selected = []
        for c in range(k):
            mask = km.labels_ == c
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue
            embs    = embeddings[mask]
            dists   = np.linalg.norm(embs - km.cluster_centers_[c], axis=1)
            best    = idxs[int(np.argmin(dists))]
            selected.append((starts[best], texts[best]))

        selected.sort(key=lambda x: x[0])
        text = " ".join(t for _, t in selected)
        text = re.sub(r"\s+", " ", text).strip()
        return (text[0].upper() + text[1:])[:800] if text else ""

    except Exception:
        # Simple fallback: evenly-spaced sentences
        step = max(1, len(texts) // n)
        return " ".join(texts[i] for i in range(0, len(texts), step))[:500]


# ═══════════════════════════════════════════════════════════════════════════════
#  Scoring
# ═══════════════════════════════════════════════════════════════════════════════

_STOP = {
    "this","that","with","have","from","they","will","been","were","their",
    "what","when","which","your","also","more","about","would","there",
    "after","before","while","other","some","just","into","only","than",
    "then","over","such","even","much","very","well","here","where",
    "being","every","both","each","many","does","must","could","might",
}
_KEY_PHRASES = [
    "most important","key point","key insight","in summary","to summarize",
    "in conclusion","the main","essentially","the point is","what this means",
    "the answer","the solution","the problem","the secret","the truth",
    "never before","revolutionary","breakthrough","very important",
    "remember this","pay attention","take note","the key","what you need",
    "you should know","most people","the reason why","that is why",
    "this is why","as a result","therefore","in fact","the best way",
    "step by step","number one","first of all","above all","critical",
    "breaking","exclusive","warning","major","biggest",
]


def score_speech(segs, audio_feat, title="", description="", tags="",
                 target=90.0) -> Tuple[List[Dict], float]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    segs = [s for s in segs if len(s["text"].split()) >= 3 and s.get("duration", 0) >= 1.0]
    if not segs:
        raise ValueError("No usable transcript segments.")

    texts = [s["text"] for s in segs]
    n     = len(texts)

    tfidf = np.asarray(
        TfidfVectorizer(stop_words="english", sublinear_tf=True,
                        ngram_range=(1, 2), max_features=12000)
        .fit_transform(texts).mean(axis=1)).flatten()

    ref   = " ".join(filter(None, [title, description[:1500], tags])).lower()
    refs  = set(re.findall(r"\b[a-z]{4,}\b", ref)) - _STOP
    kw    = np.zeros(n)
    if refs:
        for i, t in enumerate(texts):
            kw[i] = len(set(re.findall(r"\b[a-z]{4,}\b", t.lower())) & refs) / (len(refs) ** 0.5)

    kp      = np.array([sum(1.0 for p in _KEY_PHRASES if p in t.lower()) for t in texts])
    density = np.array([len(s["text"].split()) / max(s.get("duration", 2.0), 0.5) for s in segs], dtype=float)
    wf      = Counter(w for t in texts for w in re.findall(r"\b[a-z]{4,}\b", t.lower()))
    cw      = {w for w, c in wf.most_common(100) if n * 0.05 <= c <= n * 0.45}
    central = np.array([len(set(re.findall(r"\b[a-z]{4,}\b", t.lower())) & cw) for t in texts], dtype=float)

    rms = audio_feat.get("rms", np.array([]))
    aud = np.array([float(rms[int(s["start"])]) if len(rms) > int(s["start"]) else 0.0 for s in segs])

    has_meta = bool(refs)
    w = {"tfidf": 0.30 if has_meta else 0.45, "kw": 0.22 if has_meta else 0.0,
         "kp": 0.12, "density": 0.12, "central": 0.09,
         "aud": 0.15 if rms.size > 0 else 0.0}
    s = sum(w.values())
    w = {k: v / s for k, v in w.items()}

    combined = (w["tfidf"] * _minmax(tfidf) + w["kw"] * _minmax(kw) +
                w["kp"] * _minmax(kp) + w["density"] * _minmax(density) +
                w["central"] * _minmax(central) + w["aud"] * _minmax(aud))

    for i, seg in enumerate(segs):
        seg["score"] = float(combined[i])

    ranked   = sorted(segs, key=lambda x: x["score"], reverse=True)
    selected = []
    total    = 0.0
    for seg in ranked:
        dur = seg.get("duration", 2.0)
        if dur < 2.0:
            continue
        if total + dur <= target:
            selected.append(seg); total += dur
        if total >= target:
            break
    if not selected:
        selected = ranked[:15]

    selected.sort(key=lambda x: x["start"])
    return _merge_clips(selected, 3.0), sum(s.get("duration", 2.0) for s in _merge_clips(selected, 3.0))


def score_music(audio_feat: dict, video_dur: float,
                target: float = 90.0) -> Tuple[List[Dict], float]:
    n          = max(int(video_dur) + 1, 10)
    beat_times = audio_feat.get("beat_times", np.array([]))

    def pad(key):
        arr = audio_feat.get(key, np.array([]))
        if len(arr) == 0: return np.zeros(n)
        out = np.zeros(n); out[:min(len(arr), n)] = arr[:min(len(arr), n)]; return out

    rms   = pad("rms"); onset = pad("onset_env")
    has_a = rms.max() > 0 or onset.max() > 0

    if not has_a:
        return _uniform_sample(video_dur, clip_dur=8.0, target=target, margin=0.07)

    energy = 0.45 * _minmax(rms) + 0.35 * _minmax(onset) + 0.20 * np.zeros(n)
    energy = _smooth(energy, w=9)
    skip   = max(3, int(video_dur * 0.07))
    energy[:skip] = 0; energy[max(0, n - skip):] = 0

    CLIP = 8.0; used = np.zeros(n, bool); selected = []; total = 0.0
    for c in sorted(range(n), key=lambda i: -float(energy[i])):
        start = _snap_to_beat(max(0.0, float(c) - CLIP / 2), beat_times)
        end   = min(start + CLIP, video_dur - 0.5)
        dur   = end - start
        if dur < 2.5: continue
        si, ei = int(start), min(int(end) + 1, n)
        if used[si:ei].any(): continue
        selected.append({"start": start, "duration": dur})
        used[si:ei] = True; total += dur
        if total >= target: break

    if not selected:
        return _uniform_sample(video_dur, 8.0, target, 0.07)
    selected.sort(key=lambda x: x["start"])
    merged = _merge_clips(selected, 2.5)
    return merged, sum(c["duration"] for c in merged)


def score_short(dur, segs, audio_feat, title, description, tags):
    if dur <= 60:
        return [{"start": 0.0, "duration": float(dur)}], float(dur)
    target = min(30.0, dur * 0.45)
    tq = _transcript_quality(segs)
    if tq >= 0.30 and segs:
        try:
            return score_speech(segs, audio_feat, title, description, tags, target)
        except Exception:
            pass
    result = score_music(audio_feat, dur, target)
    if result[0]:
        return result
    mid = dur / 2
    s   = max(0.0, mid - target / 2)
    return [{"start": s, "duration": min(target, dur - s)}], min(target, dur - s)


def score_mixed(segs, audio_feat, title, description, tags, dur, target=90.0):
    if _transcript_quality(segs) >= 0.40 and segs:
        try:
            return score_speech(segs, audio_feat, title, description, tags, target)
        except Exception:
            pass
    return score_music(audio_feat, dur, target)


def _merge_clips(clips, gap=3.0):
    if not clips: return clips
    merged = [dict(clips[0])]
    for c in clips[1:]:
        last = merged[-1]
        if c["start"] - (last["start"] + last.get("duration", 2.0)) <= gap:
            merged[-1]["duration"] = (c["start"] + c.get("duration", 2.0)) - last["start"]
        else:
            merged.append(dict(c))
    return merged


def _uniform_sample(dur, clip_dur=8.0, target=90.0, margin=0.07):
    m = dur * margin; n = max(1, int(target / clip_dur))
    step = max(clip_dur, (dur - 2 * m) / n)
    clips = []; total = 0.0; t = m
    while t < dur - m - 1.0 and total < target:
        d = min(clip_dur, dur - m - t - 0.3)
        if d < 1.0: break
        clips.append({"start": t, "duration": d}); total += d; t += step
    return clips, total


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Analyze  (no video download)
# ═══════════════════════════════════════════════════════════════════════════════

def run_analyze(url: str, job_id: str, log_q: queue.Queue):
    def emit(msg): log_q.put({"type": "log", "msg": msg})
    def emit_meta(d): log_q.put({"type": "meta", **d})

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import NoTranscriptFound
        import yt_dlp as ytdlp

        vid = _get_video_id(url)
        if not vid:
            raise ValueError(f"Could not extract video ID from URL: {url}")

        jobs[job_id].update({"vid": vid, "url": url})
        emit(f"🎯 Video ID: {vid}")

        ffmpeg_exe = _get_ffmpeg()
        ffmpeg_dir = os.path.dirname(os.path.abspath(ffmpeg_exe))
        jobs[job_id].update({"ffmpeg_exe": ffmpeg_exe, "ffmpeg_dir": ffmpeg_dir})

        # ── Step 1: transcript + metadata (parallel) ───────────────────────
        emit("🔍 Fetching transcript & metadata in parallel...")
        tr, meta = {}, {}

        def fetch_tr():
            try:
                # Pass cookies when available (bypasses YouTube cloud-IP blocks)
                ytt_kwargs = {}
                if _HAS_COOKIES and os.path.exists(_COOKIE_FILE):
                    try:
                        ytt_kwargs["cookies"] = _COOKIE_FILE
                    except Exception:
                        pass
                ytt = YouTubeTranscriptApi(**ytt_kwargs)
                try:
                    f = ytt.fetch(vid)
                except NoTranscriptFound:
                    lst   = ytt.list(vid)
                    first = next(iter(lst))
                    emit(f"   Using: {first.language} ({first.language_code})")
                    f = first.fetch()
                tr["segs"] = [{"text": s.text, "start": s.start, "duration": s.duration} for s in f]
            except Exception as e:
                err = str(e)
                if "IP" in err or "blocked" in err or "Sign in" in err or "bot" in err:
                    emit("   ⚠️ YouTube blocked this IP (cloud provider restriction).")
                    if not _HAS_COOKIES:
                        emit("   👉 Add YOUTUBE_COOKIES secret in HF Space settings to fix this.")
                else:
                    emit(f"   No transcript: {err[:120]}")
                tr["segs"] = []

        def fetch_meta():
            try:
                with ytdlp.YoutubeDL({**_YDL_BASE,
                                       "skip_download": True, "ffmpeg_location": ffmpeg_dir}) as ydl:
                    info = ydl.extract_info(url, download=False)
                meta.update({
                    "title":       info.get("title", ""),
                    "description": (info.get("description") or "")[:1500],
                    "tags":        " ".join((info.get("tags") or [])[:30]),
                    "duration":    info.get("duration", 0),
                })
                emit_meta({
                    "title":       info.get("title", ""),
                    "channel":     info.get("uploader") or info.get("channel", "Unknown"),
                    "channel_url": info.get("uploader_url") or info.get("channel_url", ""),
                    "thumbnail":   info.get("thumbnail") or f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
                    "duration":    info.get("duration", 0),
                    "view_count":  info.get("view_count", 0),
                    "upload_date": info.get("upload_date", ""),
                    "like_count":  info.get("like_count") or 0,
                    "video_url":   f"https://www.youtube.com/watch?v={vid}",
                })
            except Exception as e:
                err = str(e)
                if "Sign in" in err or "bot" in err or "IP" in err or "blocked" in err:
                    emit("   ⚠️ YouTube is blocking this cloud IP.")
                    if not _HAS_COOKIES:
                        emit("   👉 Fix: add YOUTUBE_COOKIES secret in HF Space → Settings → Secrets.")
                else:
                    emit(f"   Metadata unavailable: {err[:120]}")
                meta.update({"title": "", "description": "", "tags": "", "duration": 0})

        t1 = threading.Thread(target=fetch_tr,   daemon=True)
        t2 = threading.Thread(target=fetch_meta, daemon=True)
        t1.start(); t2.start(); t1.join(); t2.join()

        segs    = tr.get("segs", [])
        title   = meta.get("title", "")
        vid_dur = float(meta.get("duration") or 120)
        tq      = _transcript_quality(segs)
        emit(f"✅ Got {len(segs)} transcript segments  |  \"{title}\"")

        # ── Quick type detection from metadata ─────────────────────────────
        vtype = detect_video_type(title, meta.get("description",""), meta.get("tags",""),
                                   vid_dur, 1.78, tq, url=url)

        # ── Step 2: audio analysis (only for music/mixed) ──────────────────
        audio_feat: dict = {}
        if vtype in ("music", "mixed") or (vtype == "short" and tq < 0.30):
            emit("🎵 Downloading audio stream for beat/energy analysis...")
            os.makedirs("output", exist_ok=True)
            audio_base = os.path.abspath(f"output/{vid}_audio")
            wav = download_audio_for_analysis(url, audio_base, ffmpeg_dir, max_sec=300)
            if wav:
                audio_feat.update(analyze_audio(wav))
                try: os.remove(wav)
                except Exception: pass
                emit(f"   🎵 {audio_feat.get('tempo',0):.0f} BPM · {audio_feat.get('duration',0):.0f}s analyzed")
            else:
                emit("   ⚠ Audio download failed — using uniform selection")

        # ── Step 3: Score & select clips ───────────────────────────────────
        emit(f"🧠 Scoring with {vtype.upper()}-mode analysis...")
        t0 = time.time()
        try:
            if vtype == "music":
                selected, total = score_music(audio_feat, vid_dur)
            elif vtype == "short":
                selected, total = score_short(vid_dur, segs, audio_feat,
                                              title, meta.get("description",""), meta.get("tags",""))
            elif vtype == "speech":
                selected, total = score_speech(segs, audio_feat, title,
                                               meta.get("description",""), meta.get("tags",""))
            else:
                selected, total = score_mixed(segs, audio_feat, title,
                                              meta.get("description",""), meta.get("tags",""), vid_dur)
        except Exception as e:
            emit(f"   ⚠ Scoring fallback: {e}")
            selected, total = _uniform_sample(vid_dur, 8.0, min(90.0, vid_dur * 0.6))

        emit(f"✅ Selected {len(selected)} clips → {total:.1f}s  ({time.time()-t0:.1f}s)")

        # ── Step 4: Semantic summary ───────────────────────────────────────
        summary = ""
        if segs and tq >= 0.25:
            emit("📝 Generating semantic summary (sentence-transformers + clustering)...")
            t0 = time.time()
            summary = generate_summary(segs, title=title, n=3)
            emit(f"✅ Summary generated ({time.time()-t0:.1f}s)")
        elif not segs:
            emit("   ℹ No transcript — summary unavailable for this video")

        # Save state for phase 2
        jobs[job_id].update({
            "state":      "analyzed",
            "selected":   selected,
            "total":      total,
            "title":      title,
            "meta":       meta,
            "video_type": vtype,
            "summary":    summary,
        })

        log_q.put({
            "type":       "analyzed",
            "video_type": vtype,
            "clips":      len(selected),
            "duration":   round(total, 1),
            "title":      title,
            "summary":    summary,
        })
        # NOTE: do NOT put None here — queue stays open for phase 2

    except Exception as exc:
        jobs[job_id]["state"] = "error"
        log_q.put({"type": "error", "msg": str(exc)})
        log_q.put(None)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Build  (download only selected clips via download_ranges)
# ═══════════════════════════════════════════════════════════════════════════════

def cut_concat_ffmpeg(video_path: str, segments: List[Dict],
                      output_path: str, ffmpeg_exe: str, emit_fn) -> str:
    """Stream-copy selected segments and concat — lossless, fast, always works."""
    tmp   = tempfile.mkdtemp(prefix="aivid_")
    clips = []
    try:
        emit_fn(f"✂️  Cutting {len(segments)} clips (stream-copy, no quality loss)...")
        for i, seg in enumerate(segments):
            start = max(0.0, float(seg["start"]))
            dur   = max(0.5, float(seg.get("duration", 3.0)))
            out   = os.path.join(tmp, f"clip_{i:04d}.mp4")
            r = subprocess.run([
                ffmpeg_exe, "-y",
                "-ss", f"{start:.3f}", "-i", video_path,
                "-t",  f"{dur:.3f}", "-c", "copy",
                "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", out,
            ], capture_output=True, timeout=120)
            if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 512:
                clips.append(out)
            else:
                emit_fn(f"   ⚠ Clip {i+1} skipped")
        if not clips:
            raise RuntimeError("All FFmpeg cuts failed — check that the video file exists.")
        emit_fn(f"✅ {len(clips)} clips cut — merging...")
        if len(clips) == 1:
            shutil.copy2(clips[0], output_path)
        else:
            lst = os.path.join(tmp, "list.txt")
            with open(lst, "w", encoding="utf-8") as f:
                for p in clips:
                    f.write(f"file '{p.replace(chr(92), '/')}'\n")
            r = subprocess.run([
                ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                "-i", lst, "-c", "copy", "-movflags", "+faststart", output_path,
            ], capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"FFmpeg concat error:\n{r.stderr[-400:]}")
        return output_path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_build(job_id: str, log_q: queue.Queue):
    def emit(msg): log_q.put({"type": "log", "msg": msg})

    try:
        import yt_dlp as ytdlp

        job        = jobs[job_id]
        url        = job["url"]
        vid        = job["vid"]
        selected   = job["selected"]
        ffmpeg_exe = job["ffmpeg_exe"]
        ffmpeg_dir = job["ffmpeg_dir"]
        jobs[job_id]["state"] = "building"

        os.makedirs("output", exist_ok=True)
        raw_path = os.path.abspath(f"output/{vid}_raw.mp4")
        jobs[job_id]["raw_path"] = raw_path

        # ── Download full video (cached after first run) ───────────────────
        if os.path.exists(raw_path) and os.path.getsize(raw_path) > 1_000_000:
            emit("✅ Using cached video — skipping download")
        else:
            emit("⬇️  Downloading video (best quality)...")
            last_pct = [0]

            def hook(d):
                if d["status"] == "downloading":
                    raw = d.get("_percent_str", "")
                    pct_str = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
                    spd_str = re.sub(r"\x1b\[[0-9;]*m", "",
                                     d.get("_speed_str", "")).strip()
                    eta_str = re.sub(r"\x1b\[[0-9;]*m", "",
                                     d.get("_eta_str",   "")).strip()
                    try:
                        pct = float(pct_str.replace("%", "").strip())
                        if pct - last_pct[0] >= 10:
                            last_pct[0] = pct
                            emit(f"   📥 {pct_str}  {spd_str}  ETA {eta_str}")
                    except Exception:
                        pass
                elif d["status"] == "finished":
                    emit("   ✅ Download finished, merging streams...")

            ydl_opts = {
                **_YDL_BASE,
                "format":              "bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
                "outtmpl":             raw_path,
                "ffmpeg_location":     ffmpeg_dir,
                "progress_hooks":      [hook],
            }
            with ytdlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            emit("✅ Download complete")

        # ── Cut + concat with FFmpeg (fast, lossless) ─────────────────────
        out_path = os.path.abspath(f"output/{job_id}_final.mp4")
        t0 = time.time()
        cut_concat_ffmpeg(raw_path, selected, out_path, ffmpeg_exe, emit)
        cut_time = time.time() - t0

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        total   = job.get("total", 0)
        emit(f"🎉 Done in {cut_time:.1f}s — {total:.1f}s · {len(selected)} scenes · {size_mb:.1f}MB")

        jobs[job_id].update({"state": "done", "output": out_path})
        log_q.put({
            "type":       "done",
            "duration":   round(total, 1),
            "clips":      len(selected),
            "has_raw":    True,
            "title":      job.get("title", ""),
            "size_mb":    round(size_mb, 1),
            "video_type": job.get("video_type", "speech"),
            "summary":    job.get("summary", ""),
        })

    except Exception as exc:
        jobs[job_id]["state"] = "error"
        log_q.put({"type": "error", "msg": str(exc)})
    finally:
        log_q.put(None)


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP byte-range streaming
# ═══════════════════════════════════════════════════════════════════════════════

def _stream_video(path: str) -> Response:
    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    size = os.path.getsize(path)
    rng  = request.headers.get("Range")
    if rng:
        p = rng.replace("bytes=", "").split("-")
        s = int(p[0]) if p[0] else 0
        e = int(p[1]) if len(p) > 1 and p[1] else size - 1
        e = min(e, size - 1); ln = e - s + 1
        def gen():
            with open(path, "rb") as f:
                f.seek(s); rem = ln
                while rem > 0:
                    c = f.read(min(65536, rem))
                    if not c: break
                    rem -= len(c); yield c
        return Response(gen(), 206, headers={
            "Content-Range": f"bytes {s}-{e}/{size}",
            "Accept-Ranges": "bytes", "Content-Length": str(ln), "Content-Type": "video/mp4"})
    def gf():
        with open(path, "rb") as f:
            while True:
                c = f.read(65536)
                if not c: break
                yield c
    return Response(gf(), 200, headers={
        "Accept-Ranges": "bytes", "Content-Length": str(size), "Content-Type": "video/mp4"})


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    job_id = uuid.uuid4().hex[:10]
    q: queue.Queue = queue.Queue()
    jobs[job_id] = {"state": "analyzing", "output": None, "vid": None,
                    "url": url, "queue": q}
    threading.Thread(target=run_analyze, args=(url, job_id, q), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/build/<job_id>", methods=["POST"])
def build(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    if jobs[job_id].get("state") != "analyzed":
        return jsonify({"error": "Analyze must complete first"}), 400
    q = jobs[job_id]["queue"]
    threading.Thread(target=run_build, args=(job_id, q), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/progress/<job_id>")
def progress(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    def generate():
        q = jobs[job_id]["queue"]
        while True:
            item = q.get()
            if item is None: break
            yield f"data: {json.dumps(item)}\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/video/<job_id>")
def video(job_id):
    return _stream_video(jobs.get(job_id, {}).get("output"))


@app.route("/download/<job_id>/<which>")
def download(job_id, which):
    job = jobs.get(job_id)
    if not job: return jsonify({"error": "Not found"}), 404
    path  = job.get("output")
    fname = "highlight.mp4"
    if not path or not os.path.exists(path): return jsonify({"error": "Not ready"}), 404
    return send_file(path, as_attachment=True, download_name=fname, mimetype="video/mp4")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))   # 7860 on HF Spaces, 5000 locally
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
