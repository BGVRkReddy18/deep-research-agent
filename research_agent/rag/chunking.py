"""Split documents into overlapping, paragraph-aware chunks."""

from __future__ import annotations

import re

from ..models import Chunk, Document

_PARA_SPLIT = re.compile(r"\n\s*\n+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _pieces(text: str, size: int) -> list[str]:
    """Paragraphs, with oversized paragraphs broken into sentences (and hard-split if needed)."""
    out: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= size:
            out.append(para)
            continue
        for sent in _SENT_SPLIT.split(para):
            while len(sent) > size:
                out.append(sent[:size])
                sent = sent[size:]
            if sent:
                out.append(sent)
    return out


def chunk_text(text: str, size: int = 1800, overlap: int = 250) -> list[str]:
    """Greedy packing of pieces up to `size` chars, carrying ~`overlap` chars of context forward."""
    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size // 2))
    chunks: list[str] = []
    current = ""
    for piece in _pieces(text, size):
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]  # start the overlap on a word boundary
            current = f"{tail} {piece}".strip() if tail and len(tail) + len(piece) < size else piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(docs: list[Document], prefix: str, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d_idx, doc in enumerate(docs, start=1):
        for c_idx, text in enumerate(chunk_text(doc.text, size, overlap), start=1):
            chunks.append(Chunk(id=f"{prefix}-d{d_idx}-c{c_idx}", url=doc.url, title=doc.title, text=text))
    return chunks
