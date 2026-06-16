"""RAG preprocessing engine for document cleaning, normalization, and enrichment.

Applies universal and file-type-specific cleaning rules to optimize content
for embedding-based retrieval.
"""

import re
import json as json_lib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter

import pandas as pd

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A detected section within a document."""
    title: str
    text: str
    level: int = 1  # hierarchy depth: 1=top, 2=sub, 3=paragraph
    index: int = 0


@dataclass
class ProcessedDocument:
    """Result of the preprocessing pipeline."""
    raw_text: str
    cleaned_text: str
    file_type: str
    source_file: str
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Universal cleaning helpers
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)^page\s+\d+\s*(of\s+\d+)?$"),
    re.compile(r"(?i)^©\s*.+$"),
    re.compile(r"(?i)^copyright\s*.+$"),
    re.compile(r"(?i)^all\s+rights\s+reserved\.?$"),
    re.compile(r"(?i)^confidential.*$"),
]

_DECORATIVE_LINE = re.compile(r"^[\s=\-_*~#]{4,}$")

# Common mojibake replacements
_ENCODING_FIXES = {
    "\u2018": "'", "\u2019": "'",  # smart single quotes
    "\u201c": '"', "\u201d": '"',  # smart double quotes
    "\u2013": "-", "\u2014": "-",  # en/em dash
    "\u2026": "...",               # ellipsis
    "\u00a0": " ",                 # non-breaking space
    "\ufeff": "",                  # BOM
    "\u00ad": "",                  # soft hyphen
    "\u200b": "",                  # zero-width space
}


def _fix_encoding(text: str) -> str:
    """Replace common encoding artifacts."""
    for bad, good in _ENCODING_FIXES.items():
        text = text.replace(bad, good)
    return text


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    # Collapse runs of spaces/tabs within lines
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_boilerplate(lines: List[str]) -> List[str]:
    """Remove lines that match boilerplate patterns."""
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if any(pat.match(stripped) for pat in _BOILERPLATE_PATTERNS):
            continue
        if _DECORATIVE_LINE.match(stripped):
            continue
        cleaned.append(line)
    return cleaned


def _deduplicate_paragraphs(text: str) -> str:
    """Remove exact-duplicate consecutive paragraphs."""
    paragraphs = text.split("\n\n")
    deduped = []
    prev = None
    for para in paragraphs:
        normalized = para.strip()
        if normalized and normalized != prev:
            deduped.append(para)
        prev = normalized
    return "\n\n".join(deduped)


def _extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Extract top keywords using word frequency (no external deps)."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "and", "or",
        "but", "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "some", "such", "no",
        "only", "same", "than", "too", "very", "just", "because", "as", "if",
        "when", "while", "of", "at", "by", "for", "with", "about", "against",
        "between", "through", "during", "before", "after", "above", "below",
        "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "where", "how",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "it", "its", "he", "she", "they", "them", "we", "us", "you", "i",
        "my", "your", "his", "her", "their", "our",
    }
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    counts = Counter(w for w in words if w not in stop_words)
    return [w for w, _ in counts.most_common(top_n)]


# ---------------------------------------------------------------------------
# Universal cleaning pipeline
# ---------------------------------------------------------------------------

def clean_universal(text: str) -> str:
    """Apply universal cleaning rules to any text."""
    text = _fix_encoding(text)
    lines = text.split("\n")
    lines = _remove_boilerplate(lines)
    text = "\n".join(lines)
    text = _normalize_whitespace(text)
    text = _deduplicate_paragraphs(text)
    return text


# ---------------------------------------------------------------------------
# File-type specific cleaners
# ---------------------------------------------------------------------------

def clean_pdf(text: str) -> str:
    """PDF-specific cleaning: rejoin hyphenated words, remove page markers."""
    # Rejoin hyphenated line breaks: "infor-\nmation" → "information"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Remove [Page N] markers but keep the content
    text = re.sub(r"\[Page\s+\d+\]", "", text)
    # Collapse single newlines that are just line-wrapping (not paragraph breaks)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return clean_universal(text)


def clean_txt(text: str) -> str:
    """TXT-specific cleaning: normalize bullets, detect sections."""
    # Normalize bullet characters
    text = re.sub(r"^[\s]*[•\-–—\*]\s+", "• ", text, flags=re.MULTILINE)
    # Remove ASCII art separators
    text = re.sub(r"^[=\-_\*~]{5,}\s*$", "", text, flags=re.MULTILINE)
    return clean_universal(text)


def clean_csv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a CSV/XLSX DataFrame: normalize columns, drop empties."""
    # Normalize column names to snake_case
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
        for col in df.columns
    ]
    # Drop fully empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")
    # Fill NaN with empty string for text processing
    df = df.fillna("")
    return df


def dataframe_to_sentences(df: pd.DataFrame, sheet_name: str = None) -> List[Dict[str, Any]]:
    """Convert DataFrame rows to natural-language sentences with metadata.

    Returns a list of dicts: {"text": "...", "row_number": N, "sheet_name": "..."}
    """
    columns = list(df.columns)
    results = []

    for row_idx, row in df.iterrows():
        parts = []
        for col in columns:
            val = str(row[col]).strip()
            if val and val.lower() not in ("", "nan", "none", "null"):
                col_label = col.replace("_", " ").title()
                parts.append(f"{col_label} is {val}")

        if parts:
            sentence = "; ".join(parts) + "."
            meta = {"row_number": int(row_idx) + 1}
            if sheet_name:
                meta["sheet_name"] = sheet_name
            results.append({"text": sentence, **meta})

    return results


def clean_json_data(data: Any, prefix: str = "") -> List[str]:
    """Flatten JSON into natural-language statements."""
    lines = []

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if value is None or value == "" or value == []:
                continue
            if isinstance(value, (dict, list)):
                lines.extend(clean_json_data(value, full_key))
            else:
                label = full_key.replace("_", " ").replace(".", " > ").title()
                lines.append(f"{label}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            lines.extend(clean_json_data(item, f"{prefix}[{i}]"))
    else:
        if data is not None and str(data).strip():
            label = prefix.replace("_", " ").replace(".", " > ").title() if prefix else "Value"
            lines.append(f"{label}: {data}")

    return lines


def clean_yaml_data(data: Any) -> List[str]:
    """Flatten YAML data into natural-language statements (same as JSON)."""
    return clean_json_data(data)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    (re.compile(r"^#+\s+(.+)$"), 1),                    # Markdown: # Heading
    (re.compile(r"^(\d+\.)+\s+(.+)$"), 2),               # Numbered: 1.2 Heading
    (re.compile(r"^[A-Z][A-Z\s]{5,}$"), 1),              # ALL CAPS heading
    (re.compile(r"^(Chapter|Section|Part)\s+\d+", re.I), 1),  # Chapter/Section N
]


def detect_sections(text: str) -> List[Section]:
    """Detect section boundaries from headings in text."""
    lines = text.split("\n")
    sections = []
    current_title = "Introduction"
    current_lines = []
    current_level = 1
    section_idx = 0

    for line in lines:
        matched = False
        for pattern, level in _HEADING_PATTERNS:
            m = pattern.match(line.strip())
            if m:
                # Save previous section
                if current_lines:
                    sections.append(Section(
                        title=current_title,
                        text="\n".join(current_lines).strip(),
                        level=current_level,
                        index=section_idx,
                    ))
                    section_idx += 1
                # Start new section
                current_title = m.group(m.lastindex) if m.lastindex else line.strip()
                current_title = current_title.strip("# ").strip()
                current_level = level
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    # Final section
    if current_lines:
        sections.append(Section(
            title=current_title,
            text="\n".join(current_lines).strip(),
            level=current_level,
            index=section_idx,
        ))

    # If no sections detected, treat entire document as one section
    if not sections:
        sections.append(Section(
            title="Document Content",
            text=text,
            level=1,
            index=0,
        ))

    return sections


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_document(
    raw_text: str,
    file_type: str,
    source_file: str,
    dataframe: pd.DataFrame = None,
    parsed_data: Any = None,
    sheet_dataframes: Dict[str, pd.DataFrame] = None,
) -> ProcessedDocument:
    """
    Run the full preprocessing pipeline on a document.

    Args:
        raw_text: Raw extracted text (for PDF/TXT)
        file_type: File extension (e.g., ".pdf", ".csv")
        source_file: Original filename
        dataframe: Parsed DataFrame (for CSV)
        parsed_data: Parsed dict/list (for JSON/YAML)
        sheet_dataframes: Dict of sheet_name → DataFrame (for XLSX)

    Returns:
        ProcessedDocument with cleaned text, sections, and metadata
    """
    logger.info("Preprocessing document", file_type=file_type, source_file=source_file)

    cleaned_text = ""
    sections = []
    doc_metadata = {
        "source_file": source_file,
        "file_type": file_type,
    }

    if file_type == ".pdf":
        cleaned_text = clean_pdf(raw_text)
        sections = detect_sections(cleaned_text)

    elif file_type == ".txt":
        cleaned_text = clean_txt(raw_text)
        sections = detect_sections(cleaned_text)

    elif file_type == ".csv" and dataframe is not None:
        df = clean_csv_dataframe(dataframe)
        rows = dataframe_to_sentences(df)
        cleaned_text = "\n".join(r["text"] for r in rows)
        doc_metadata["column_names"] = list(df.columns)
        doc_metadata["row_count"] = len(df)
        # Group into a single section
        sections = [Section(
            title=f"CSV Data: {source_file}",
            text=cleaned_text,
            level=1,
            index=0,
        )]

    elif file_type in (".xlsx", ".xls") and sheet_dataframes:
        all_text_parts = []
        for sheet_name, df in sheet_dataframes.items():
            df = clean_csv_dataframe(df)
            rows = dataframe_to_sentences(df, sheet_name=sheet_name)
            sheet_text = "\n".join(r["text"] for r in rows)
            all_text_parts.append(sheet_text)
            sections.append(Section(
                title=f"Sheet: {sheet_name}",
                text=sheet_text,
                level=1,
                index=len(sections),
            ))
        cleaned_text = "\n\n".join(all_text_parts)
        doc_metadata["sheet_names"] = list(sheet_dataframes.keys())

    elif file_type == ".json" and parsed_data is not None:
        statements = clean_json_data(parsed_data)
        cleaned_text = "\n".join(statements)
        cleaned_text = clean_universal(cleaned_text)
        sections = [Section(
            title=f"JSON Data: {source_file}",
            text=cleaned_text,
            level=1,
            index=0,
        )]

    elif file_type in (".yaml", ".yml") and parsed_data is not None:
        statements = clean_yaml_data(parsed_data)
        cleaned_text = "\n".join(statements)
        cleaned_text = clean_universal(cleaned_text)
        sections = [Section(
            title=f"YAML Data: {source_file}",
            text=cleaned_text,
            level=1,
            index=0,
        )]

    else:
        # Fallback: apply universal cleaning to raw text
        cleaned_text = clean_universal(raw_text)
        sections = detect_sections(cleaned_text)

    doc_metadata["section_count"] = len(sections)
    doc_metadata["keywords"] = _extract_keywords(cleaned_text)

    logger.info(
        "Preprocessing complete",
        source_file=source_file,
        sections=len(sections),
        cleaned_length=len(cleaned_text),
    )

    return ProcessedDocument(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        file_type=file_type,
        source_file=source_file,
        sections=sections,
        metadata=doc_metadata,
    )
