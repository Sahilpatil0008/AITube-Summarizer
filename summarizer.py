"""
summarizer.py — Generate an abstractive summary with BART and rank every
transcript segment by semantic proximity to that summary using
sentence-transformers (all-MiniLM-L6-v2).
"""
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)

BART_MODEL = "facebook/bart-large-cnn"
SBERT_MODEL = "all-MiniLM-L6-v2"
# Stay well within BART's 1024-token context (≈ 3000 chars ≈ 750 tokens)
CHUNK_CHARS = 3000


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_into_chunks(text: str, max_chars: int = CHUNK_CHARS) -> List[str]:
    """Split *text* at sentence boundaries so each chunk fits under *max_chars*."""
    sentences = text.replace(". ", ".\n").split("\n")
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if length + len(sent) > max_chars and current:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(sent)
        length += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks or [text[:max_chars]]


# ── public API ────────────────────────────────────────────────────────────────

def summarize_transcript(
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarise the full transcript and attach a *semantic_score* to every segment.

    Args:
        segments: List of {start, end, text} dicts produced by transcriber.py.

    Returns:
        {
            "summary":          str — BART-generated abstractive summary,
            "ranked_segments":  list sorted by semantic_score (desc),
            "all_segments":     original unsorted list,
        }
    """
    full_text = " ".join(s["text"] for s in segments)
    logger.info(
        "Transcript: %d chars across %d segments.", len(full_text), len(segments)
    )

    # ── BART abstractive summarisation ────────────────────────────────────────
    logger.info("Loading BART model (%s)...", BART_MODEL)
    tokenizer = BartTokenizer.from_pretrained(BART_MODEL)
    model = BartForConditionalGeneration.from_pretrained(BART_MODEL)
    model.eval()

    chunks = _split_into_chunks(full_text)
    logger.info("Summarising %d chunk(s) with BART...", len(chunks))
    summary_parts: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        logger.info("  Chunk %d/%d (%d chars)...", idx, len(chunks), len(chunk))
        inputs = tokenizer(
            chunk,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            ids = model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=40,
                num_beams=4,
                early_stopping=True,
            )
        summary_parts.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    summary_text = " ".join(summary_parts)
    logger.info(
        "Summary generated (%d chars): %s…", len(summary_text), summary_text[:120]
    )

    # ── Sentence-transformer semantic ranking ─────────────────────────────────
    logger.info("Loading SentenceTransformer (%s)...", SBERT_MODEL)
    sbert = SentenceTransformer(SBERT_MODEL)

    seg_texts = [s["text"] for s in segments]
    logger.info("Encoding %d segment(s)...", len(seg_texts))

    seg_embeddings = sbert.encode(
        seg_texts, show_progress_bar=False, batch_size=64, normalize_embeddings=True
    )
    sum_embedding = sbert.encode(
        [summary_text], show_progress_bar=False, normalize_embeddings=True
    )

    # cosine_similarity returns (n_segments × 1); flatten to 1-D
    similarities: np.ndarray = cosine_similarity(seg_embeddings, sum_embedding).flatten()

    ranked: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        ranked.append({**seg, "semantic_score": float(similarities[i])})

    ranked.sort(key=lambda x: x["semantic_score"], reverse=True)
    logger.info(
        "Top segment (score=%.4f): '%s'",
        ranked[0]["semantic_score"],
        ranked[0]["text"][:80],
    )

    return {
        "summary": summary_text,
        "ranked_segments": ranked,
        "all_segments": segments,
    }
