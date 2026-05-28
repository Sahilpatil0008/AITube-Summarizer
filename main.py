import sys, os
# Force UTF-8 output so emoji/Unicode prints correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_video_id(url):
    url = url.strip()
    if 'youtu.be' in url:
        return url.split('youtu.be/')[-1].split('?')[0].split('&')[0]
    parsed = urlparse(url)
    return parse_qs(parsed.query).get('v', [None])[0]

def get_transcript(video_id):
    print("📝 Fetching transcript...")
    ytt = YouTubeTranscriptApi()
    try:
        fetched = ytt.fetch(video_id)
    except NoTranscriptFound:
        # English not available — fall back to the first available language
        transcript_list = ytt.list(video_id)
        first = next(iter(transcript_list))
        print(f"   English not found, using: {first.language} ({first.language_code})")
        fetched = first.fetch()
    segs = [{'text': s.text, 'start': s.start, 'duration': s.duration} for s in fetched]
    print(f"✅ Got {len(segs)} segments")
    return segs

def score_and_select(segs, target=90):
    print("🧠 Scoring segments...")
    texts = [s['text'] for s in segs]
    vec = TfidfVectorizer(stop_words='english')
    mat = vec.fit_transform(texts).toarray()
    scores = mat.sum(axis=1)
    for i, s in enumerate(segs):
        s['score'] = float(scores[i])
    ranked = sorted(segs, key=lambda x: x['score'], reverse=True)
    selected, total = [], 0
    for s in ranked:
        dur = s.get('duration', 2.0)
        if total + dur <= target:
            selected.append(s)
            total += dur
        if total >= target:
            break
    selected.sort(key=lambda x: x['start'])
    print(f"✅ Selected {len(selected)} clips → {total:.1f}s total")
    return selected

def build_video(url, clips, video_id):
    print("⬇️  Downloading video...")
    os.makedirs("output", exist_ok=True)
    raw = f"output/{video_id}_raw.mp4"
    out = f"output/{video_id}_final.mp4"
    os.system(f'yt-dlp -f "best[ext=mp4]" -o "{raw}" "{url}"')
    print("✂️  Cutting clips...")
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    v = VideoFileClip(raw)
    parts = []
    for s in clips:
        start = s['start']
        end = min(start + s.get('duration', 2.0), v.duration)
        parts.append(v.subclip(start, end))
    final = concatenate_videoclips(parts, method="compose")
    final.write_videofile(out, codec="libx264", audio_codec="aac", logger=None)
    print(f"\n🎉 Done! → {out}  ({final.duration:.1f}s)")

if __name__ == "__main__":
    url = sys.argv[1]
    vid = get_video_id(url)
    print(f"🎯 Video ID: {vid}")
    transcript = get_transcript(vid)
    clips = score_and_select(transcript, target=90)
    build_video(url, clips, vid)