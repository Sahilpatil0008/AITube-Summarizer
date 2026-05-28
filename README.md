---
title: AITube Summarizer
emoji: 🎬
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: AI-powered YouTube highlight generator — analyze first, download only the best moments
---

# 🎬 AITube Summarizer

> **Analyze first. Download only the best moments.**

An AI-powered YouTube highlight generator that creates smart 90-second clips from any video — Speech, Music, Mixed content, and Shorts — without re-encoding, preserving original quality.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Details |
|---|---|
| **Two-phase pipeline** | Phase 1 analyzes transcript & generates summary (no download). Phase 2 downloads only selected clips. |
| **Auto video-type detection** | 🎤 Speech · 🎵 Music · 🎬 Mixed · ⚡ Shorts |
| **7-signal transcript scoring** | TF-IDF bigrams · keyword relevance · key-phrase hits · density · centrality · audio energy · visual motion |
| **Beat detection (music)** | librosa beat tracking + energy peaks — no transcript needed |
| **Semantic AI summary** | sentence-transformers `all-MiniLM-L6-v2` + KMeans clustering |
| **Lossless output** | FFmpeg stream-copy — original codec, zero quality loss |
| **YouTube-style UI** | Dark mode, two-column watch page, chip filters, action bar |
| **Shorts support** | 2-min video → 30-second highlight |
| **Video caching** | Same video analyzed twice → skips download entirely |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Sahilpatil0008/AITube-Summarizer.git
cd AITube-Summarizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
# Open http://localhost:5000
```

---

## 🧠 How It Works

```
Phase 1 — Analyze (5–15 seconds, no download)
  ├── Fetch YouTube transcript + metadata (parallel)
  ├── For music: download audio-only stream → librosa beat detection
  ├── Score segments (7 signals combined)
  ├── Select best clips
  └── Generate semantic summary (SBERT + KMeans)

Phase 2 — Build (download + assemble)
  ├── Download full video (cached after first run)
  ├── FFmpeg stream-copy cut selected segments
  └── Concatenate → final highlight MP4
```

---

## 📦 Tech Stack

- **Backend**: Flask, yt-dlp, FFmpeg
- **Audio analysis**: librosa, scipy
- **Visual analysis**: OpenCV
- **NLP / Scoring**: scikit-learn (TF-IDF), sentence-transformers, KMeans
- **Frontend**: Vanilla JS, YouTube-inspired dark UI

---

## 📁 Project Structure

```
ai-video-summarizer/
├── app.py              # Main Flask app + full pipeline
├── templates/
│   └── index.html      # YouTube-style frontend
├── main.py             # Lightweight CLI version
├── pipeline.py         # Advanced pipeline (Whisper + BART)
├── downloader.py       # yt-dlp video downloader
├── transcriber.py      # Whisper transcription
├── summarizer.py       # BART summarization
├── scorer.py           # Multi-signal scorer
├── video_cutter.py     # MoviePy clip cutter
└── requirements.txt
```

---

## 🎯 Supported URL Formats

```
https://youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://youtube.com/shorts/VIDEO_ID
```

---

## 📋 Requirements

- Python 3.10+
- ffmpeg (bundled via `imageio-ffmpeg` — no system install needed)
- Internet connection (for YouTube download)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ by [Sahil Patil](https://github.com/Sahilpatil0008)*
