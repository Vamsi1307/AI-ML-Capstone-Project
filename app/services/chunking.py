"""Data chunking service with hierarchical chunking, semantic titles, and metadata.

Supports fixed, overlap, and semantic chunking strategies.
Enriches each chunk with IDs, titles, summaries, keywords, and hierarchy metadata.
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.preprocessing import ProcessedDocument, Section, _extract_keywords

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with rich metadata for RAG retrieval."""

    text: str
    index: int
    chunk_id: str = ""
    title: str = ""
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Chunk(id='{self.chunk_id}', title='{self.title[:40]}', length={len(self.text)})"


class ChunkingService:
    """Service for splitting documents into enriched, retrievable chunks."""

    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
        strategy: str = None,
    ):
        """
        Initialize chunking service.

        Args:
            chunk_size: Size of each chunk in tokens/words (default from settings)
            overlap: Overlap between chunks in tokens (default from settings)
            strategy: Chunking strategy - 'fixed', 'overlap', 'semantic'
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP
        self.strategy = strategy or settings.CHUNKING_STRATEGY

    # ------------------------------------------------------------------
    # Main entry point: process a ProcessedDocument
    # ------------------------------------------------------------------

    def chunk_processed_document(
        self,
        doc: ProcessedDocument,
        document_id: str = None,
    ) -> List[Chunk]:
        """
        Chunk a ProcessedDocument using hierarchical, section-aware splitting.

        This is the preferred entry point. It uses detected sections for
        hierarchy-aware chunking and enriches each chunk with metadata.

        Args:
            doc: A ProcessedDocument from the preprocessing pipeline
            document_id: Override document ID (defaults to doc.source_file)

        Returns:
            List of enriched Chunk objects
        """
        doc_id = document_id or doc.source_file
        logger.info(
            "Chunking processed document",
            document_id=doc_id,
            sections=len(doc.sections),
            strategy=self.strategy,
        )

        all_chunks: List[Chunk] = []
        global_index = 0

        for section in doc.sections:
            # Split section text into raw chunks
            if self.strategy == "fixed":
                raw_chunks = self._chunk_fixed(section.text)
            elif self.strategy == "overlap":
                raw_chunks = self._chunk_overlap(section.text)
            elif self.strategy == "semantic":
                raw_chunks = self._chunk_semantic(section.text)
            else:
                raw_chunks = self._chunk_overlap(section.text)

            for chunk_idx, raw in enumerate(raw_chunks):
                chunk_id = f"{doc_id}-sec{section.index}-chunk{chunk_idx}"

                # Generate semantic title
                title = self._generate_title(raw.text, section.title, chunk_idx)

                # Generate summary
                summary = self._generate_summary(raw.text)

                # Extract keywords
                keywords = _extract_keywords(raw.text, top_n=5)

                # Build metadata
                chunk_metadata = {
                    "source_file": doc.source_file,
                    "file_type": doc.file_type,
                    "document_id": doc_id,
                    "section_title": section.title,
                    "hierarchy_level": section.level,
                    "keywords": keywords,
                    "chunk_index": global_index,
                }

                # Merge any doc-level metadata (sheet_name, row_number, etc.)
                for key in ("sheet_name", "row_number", "column_names"):
                    if key in doc.metadata:
                        chunk_metadata[key] = doc.metadata[key]

                all_chunks.append(Chunk(
                    text=raw.text,
                    index=global_index,
                    chunk_id=chunk_id,
                    title=title,
                    summary=summary,
                    metadata=chunk_metadata,
                ))
                global_index += 1

        logger.info(
            "Hierarchical chunking complete",
            document_id=doc_id,
            chunk_count=len(all_chunks),
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Backward-compatible: chunk plain text (used by existing callers)
    # ------------------------------------------------------------------

    def chunk_document(self, content: str, document_id: str = None) -> List[Chunk]:
        """
        Split plain document content into chunks (backward-compatible).

        Args:
            content: Document content to chunk
            document_id: Optional document identifier for metadata

        Returns:
            List of Chunk objects
        """
        logger.info(
            "Chunking document (plain text)",
            strategy=self.strategy,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        try:
            if self.strategy == "fixed":
                raw_chunks = self._chunk_fixed(content)
            elif self.strategy == "overlap":
                raw_chunks = self._chunk_overlap(content)
            elif self.strategy == "semantic":
                raw_chunks = self._chunk_semantic(content)
            else:
                logger.warning("Unknown strategy, defaulting to overlap")
                raw_chunks = self._chunk_overlap(content)

            # Enrich each chunk with IDs, titles, summaries
            enriched = []
            for i, raw in enumerate(raw_chunks):
                chunk_id = f"{document_id or 'doc'}-chunk{i}"
                raw.chunk_id = chunk_id
                raw.title = self._generate_title(raw.text, "Document", i)
                raw.summary = self._generate_summary(raw.text)
                raw.metadata["keywords"] = _extract_keywords(raw.text, top_n=5)
                if document_id:
                    raw.metadata["document_id"] = document_id
                enriched.append(raw)

            logger.info("Document chunking complete", chunk_count=len(enriched))
            return enriched

        except Exception as e:
            logger.error("Document chunking failed", error=str(e))
            raise

    # ------------------------------------------------------------------
    # Title and summary generation (heuristic, no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_title(text: str, section_title: str, chunk_index: int) -> str:
        """Generate a semantic title for a chunk.

        Uses the first meaningful sentence or the section title + index.
        """
        # Try to use the first sentence as a title base
        sentences = re.split(r"[.!?]\s+", text.strip())
        if sentences and len(sentences[0]) > 10:
            first = sentences[0].strip()
            # Truncate to reasonable title length
            if len(first) > 80:
                first = first[:77] + "..."
            return first

        # Fallback to section title
        if section_title and section_title != "Document Content":
            return f"{section_title} (Part {chunk_index + 1})"

        return f"Content Block {chunk_index + 1}"

    @staticmethod
    def _generate_summary(text: str, max_sentences: int = 2) -> str:
        """Generate a summary from the first N sentences of the chunk."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        summary_sentences = sentences[:max_sentences]
        summary = " ".join(s.strip() for s in summary_sentences if s.strip())
        if len(summary) > 300:
            summary = summary[:297] + "..."
        return summary

    # ------------------------------------------------------------------
    # Core splitting strategies
    # ------------------------------------------------------------------

    def _chunk_fixed(self, content: str) -> List[Chunk]:
        """Split content into fixed-size chunks by word count."""
        words = content.split()
        chunks = []
        chunk_index = 0

        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(Chunk(text=chunk_text, index=chunk_index))
            chunk_index += 1

        return chunks

    def _chunk_overlap(self, content: str) -> List[Chunk]:
        """Split content into chunks with specified overlap."""
        words = content.split()
        chunks = []
        chunk_index = 0
        stride = max(self.chunk_size - self.overlap, 1)

        for i in range(0, len(words), stride):
            chunk_words = words[i : i + self.chunk_size]
            if len(chunk_words) > 0:
                chunk_text = " ".join(chunk_words)
                chunks.append(Chunk(text=chunk_text, index=chunk_index))
                chunk_index += 1

            # Stop if we've reached the end
            if i + self.chunk_size >= len(words):
                break

        return chunks

    def _chunk_semantic(self, content: str) -> List[Chunk]:
        """Split content at semantic boundaries (sentences/paragraphs)."""
        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())

            for sentence in sentences:
                test_chunk = (
                    (current_chunk + " " + sentence).strip()
                    if current_chunk
                    else sentence
                )

                word_count = len(test_chunk.split())
                if word_count <= self.chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(Chunk(text=current_chunk, index=chunk_index))
                        chunk_index += 1
                    current_chunk = sentence

        if current_chunk:
            chunks.append(Chunk(text=current_chunk, index=chunk_index))

        return chunks
