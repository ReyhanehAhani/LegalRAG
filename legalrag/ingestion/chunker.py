"""Hierarchical chunker for legal text.

Strategy
--------
Two-level hierarchy:
  - Parent chunks  : large, semantically coherent sections (~parent_size chars),
                     stored as-is.  Used for context expansion at generation time.
  - Child chunks   : smaller overlapping windows (child_size / child_overlap)
                     that are actually embedded and indexed.

Each child chunk carries a reference (parent_chunk_id) to its parent so the
generator can retrieve the full parent for richer context when needed.

This approach follows the "small-to-big retrieval" pattern:
  retrieve child (precise vector match) → expand to parent (full context).

Implementation note
-------------------
Splitting is done entirely by *character position* in the original text so
that ``char_start`` / ``char_end`` are always exact offsets into the source
document.  We never reconstruct text by joining tokens (which loses whitespace)
and never use ``str.find()`` on reconstructed strings (which is fragile when
the original contains multi-space / newline runs).
"""

from __future__ import annotations

import logging
import re

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseChunker
from legalrag.core.models import Chunk, RawDocument, stable_id

logger = logging.getLogger(__name__)


# ── Sentence-boundary positions ───────────────────────────────────────────────

_SENT_END_RE = re.compile(r"[.!?](?=\s)")


def _sentence_end_positions(text: str) -> list[int]:
    """Return character positions *just after* each sentence-ending punctuation.

    Each value is the index of the first whitespace character after '.', '!',
    or '?'.  The caller uses these as candidate split points when building
    parent chunks.
    """
    return [m.end() for m in _SENT_END_RE.finditer(text)]


# ── Parent-level splitting (position-based) ───────────────────────────────────


def _split_positions(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans that partition *text* completely.

    Spans are at most *max_chars* wide.  We try to break at a sentence boundary
    (first ``[.!?]\\s`` after the target length) to keep coherent units, but
    always fall back to a hard cut so no character is ever skipped.

    The spans are contiguous and non-overlapping: ``spans[i][1] == spans[i+1][0]``
    and the union covers ``[0, len(text))``.
    """
    if not text:
        return []

    ends = _sentence_end_positions(text)
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)

    while start < n:
        target = start + max_chars
        if target >= n:
            spans.append((start, n))
            break

        # Find the first sentence boundary at or after `target`
        # (search within a reasonable lookahead to avoid runaway chunks)
        lookahead = target + max_chars // 2
        boundary = next(
            (e for e in ends if target <= e <= lookahead),
            None,
        )
        end = boundary if boundary is not None else target
        spans.append((start, end))
        start = end

    return spans


# ── HierarchicalChunker ───────────────────────────────────────────────────────


class HierarchicalChunker(BaseChunker):
    """Produces parent + child chunks with exact character offsets.

    Parameters
    ----------
    parent_size:
        Approximate character length of a parent chunk (default 1500).
    child_size:
        Approximate character length of a child chunk (default from config).
    child_overlap:
        Character overlap between consecutive child chunks (default from config).
    """

    def __init__(
        self,
        parent_size: int = 1500,
        child_size: int | None = None,
        child_overlap: int | None = None,
    ) -> None:
        cfg = settings.retrieval
        self._parent_size = parent_size
        self._child_size = child_size or cfg.chunk_size
        self._child_overlap = child_overlap or cfg.chunk_overlap

    def chunk(self, document: RawDocument) -> list[Chunk]:
        text = document.text
        doc_id = document.metadata.doc_id
        parent_spans = _split_positions(text, self._parent_size)

        all_chunks: list[Chunk] = []

        for p_start, p_end in parent_spans:
            parent_text = text[p_start:p_end]
            parent_id = stable_id(doc_id, str(p_start), str(p_end))

            parent_chunk = Chunk(
                chunk_id=parent_id,
                doc_id=doc_id,
                parent_chunk_id=None,
                text=parent_text,
                char_start=p_start,
                char_end=p_end,
                metadata=document.metadata,
            )
            all_chunks.append(parent_chunk)

            # Child chunks: sliding window over the parent span
            children = self._sliding_window(
                parent_text, offset=p_start, doc_id=doc_id, parent_id=parent_id,
                metadata=document.metadata,
            )
            all_chunks.extend(children)

        logger.debug(
            "Chunked '%s' → %d chunks (%d parents)",
            document.metadata.source_path,
            len(all_chunks),
            len(parent_spans),
        )
        return all_chunks

    def _sliding_window(
        self,
        text: str,
        offset: int,
        doc_id: str,
        parent_id: str,
        metadata,
    ) -> list[Chunk]:
        """Produce overlapping child chunks with deterministic IDs."""
        chunks: list[Chunk] = []
        step = max(1, self._child_size - self._child_overlap)
        pos = 0
        n = len(text)

        while pos < n:
            end = min(pos + self._child_size, n)
            char_start = offset + pos
            char_end = offset + end
            chunks.append(
                Chunk(
                    chunk_id=stable_id(doc_id, str(char_start), str(char_end)),
                    doc_id=doc_id,
                    parent_chunk_id=parent_id,
                    text=text[pos:end],
                    char_start=char_start,
                    char_end=char_end,
                    metadata=metadata,
                )
            )
            if end == n:
                break
            pos += step

        return chunks
