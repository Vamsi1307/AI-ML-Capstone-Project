"""Smoke test for the RAG preprocessing pipeline."""
import json
import pandas as pd
from app.services.preprocessing import (
    preprocess_document, clean_pdf, clean_txt,
    clean_csv_dataframe, dataframe_to_sentences,
    clean_json_data, detect_sections, _extract_keywords,
)
from app.services.chunking import ChunkingService

# --- Test 1: PDF cleaning ---
pdf_text = """[Page 1]
This is a test docu-
ment about artificial intelli-
gence and machine learning.

Page 1 of 5

Copyright 2024 Acme Inc.
All rights reserved.

[Page 2]
INTRODUCTION

AI has transformed many fields including
healthcare, finance, and education.

================

The applications are growing rapidly."""

cleaned = clean_pdf(pdf_text)
assert "document about artificial intelligence" in cleaned, f"Hyphen rejoin failed: {cleaned[:100]}"
assert "Page 1 of 5" not in cleaned, "Boilerplate not removed"
assert "Copyright" not in cleaned, "Copyright not removed"
assert "====" not in cleaned, "Decorative line not removed"
print("Test 1 PASS: PDF cleaning (hyphen rejoin, boilerplate, decorative)")

# --- Test 2: CSV to sentences ---
df = pd.DataFrame({
    "Name": ["Alice", "Bob"],
    "Age": [30, 25],
    "City": ["New York", "London"],
})
df = clean_csv_dataframe(df)
rows = dataframe_to_sentences(df)
assert len(rows) == 2
assert "Alice" in rows[0]["text"]
assert "30" in rows[0]["text"]
print(f"Test 2 PASS: CSV row-to-sentence: '{rows[0]['text']}'")

# --- Test 3: JSON flattening ---
data = {
    "user": {"name": "Alice", "age": 30},
    "settings": {"theme": "dark", "notifications": None},
}
statements = clean_json_data(data)
assert any("Alice" in s for s in statements)
assert not any("None" in s for s in statements), "Null not removed"
print(f"Test 3 PASS: JSON flattening ({len(statements)} statements)")

# --- Test 4: Section detection ---
text = """# Introduction
This is the intro section.

# Methods
We used various methods.

## Sub-methods
Including sub-methods.

# Results
Here are the results."""

sections = detect_sections(text)
assert len(sections) >= 3, f"Expected >=3 sections, got {len(sections)}"
assert sections[0].title == "Introduction"
print(f"Test 4 PASS: Section detection ({len(sections)} sections found)")

# --- Test 5: Keyword extraction ---
kw = _extract_keywords("The machine learning model trained on neural network data produces accurate predictions", 3)
assert len(kw) == 3
print(f"Test 5 PASS: Keywords extracted: {kw}")

# --- Test 6: Full pipeline + chunking ---
doc = preprocess_document(
    raw_text=text,
    file_type=".txt",
    source_file="test.txt",
)
chunker = ChunkingService(chunk_size=50, overlap=10)
chunks = chunker.chunk_processed_document(doc, document_id="test.txt")
assert len(chunks) > 0
assert chunks[0].chunk_id != ""
assert chunks[0].title != ""
assert chunks[0].summary != ""
assert "keywords" in chunks[0].metadata
print(f"Test 6 PASS: Full pipeline -> {len(chunks)} enriched chunks")
print(f"  First chunk: id='{chunks[0].chunk_id}', title='{chunks[0].title[:50]}'")
print(f"  Keywords: {chunks[0].metadata['keywords']}")

# --- Test 7: Chunk output format (JSONL-ready) ---
chunk_json = {
    "chunk_id": chunks[0].chunk_id,
    "title": chunks[0].title,
    "summary": chunks[0].summary,
    "metadata": chunks[0].metadata,
    "text": chunks[0].text[:100],
}
print(f"\nTest 7 PASS: JSONL-ready chunk:\n{json.dumps(chunk_json, indent=2)}")

print("\n[OK] All preprocessing pipeline tests passed!")
