"""Document ingestion service for various file formats.

Extracts content from files and runs the preprocessing pipeline to produce
cleaned, structured documents ready for chunking.
"""

import os
import json as json_lib
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import yaml
from pypdf import PdfReader

from app.core.logging_config import get_logger
from app.services.preprocessing import preprocess_document, ProcessedDocument

logger = get_logger(__name__)


class DocumentIngestionService:
    """Service for extracting, validating, and cleaning documents from various formats."""

    SUPPORTED_FORMATS = {".pdf", ".txt", ".csv", ".xlsx", ".xls", ".json", ".yaml", ".yml"}
    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

    @staticmethod
    def validate_file(file_path: str) -> bool:
        """
        Validate file existence and format support.

        Args:
            file_path: Path to the file to validate

        Returns:
            True if file is valid, False otherwise
        """
        path = Path(file_path)

        if not path.exists():
            logger.warning("File not found", file_path=file_path)
            return False

        if path.suffix.lower() not in DocumentIngestionService.SUPPORTED_FORMATS:
            logger.warning(
                "Unsupported file format", file_path=file_path, suffix=path.suffix
            )
            return False

        file_size = path.stat().st_size
        if file_size > DocumentIngestionService.MAX_FILE_SIZE_BYTES:
            logger.warning(
                "File size exceeds limit",
                file_path=file_path,
                size_mb=file_size / (1024 * 1024),
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Extraction methods — return structured data per file type
    # ------------------------------------------------------------------

    @staticmethod
    def extract_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        logger.info("Extracting PDF", file_path=file_path)
        text = ""
        try:
            with open(file_path, "rb") as file:
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    text += f"\n[Page {page_num + 1}]\n"
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        except Exception as e:
            logger.error("PDF extraction failed", file_path=file_path, error=str(e))
            raise
        return text

    @staticmethod
    def extract_txt(file_path: str) -> str:
        """Extract text from text file."""
        logger.info("Extracting TXT", file_path=file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            logger.error("Text extraction failed", file_path=file_path, error=str(e))
            raise

    @staticmethod
    def extract_csv(file_path: str) -> pd.DataFrame:
        """Extract content from CSV file as DataFrame."""
        logger.info("Extracting CSV", file_path=file_path)
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error("CSV extraction failed", file_path=file_path, error=str(e))
            raise

    @staticmethod
    def extract_excel(file_path: str) -> Dict[str, pd.DataFrame]:
        """Extract content from Excel file as dict of sheet_name → DataFrame."""
        logger.info("Extracting Excel", file_path=file_path)
        try:
            excel_file = pd.ExcelFile(file_path)
            sheets = {}
            for sheet_name in excel_file.sheet_names:
                sheets[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)
            return sheets
        except Exception as e:
            logger.error("Excel extraction failed", file_path=file_path, error=str(e))
            raise

    @staticmethod
    def extract_json(file_path: str) -> Any:
        """Extract content from JSON file as parsed data."""
        logger.info("Extracting JSON", file_path=file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json_lib.load(file)
        except Exception as e:
            logger.error("JSON extraction failed", file_path=file_path, error=str(e))
            raise

    @staticmethod
    def extract_yaml(file_path: str) -> Any:
        """Extract content from YAML file as parsed data."""
        logger.info("Extracting YAML", file_path=file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error("YAML extraction failed", file_path=file_path, error=str(e))
            raise

    # ------------------------------------------------------------------
    # Main ingestion method
    # ------------------------------------------------------------------

    def ingest_document(self, file_path: str, source_filename: str = None) -> ProcessedDocument:
        """
        Extract and preprocess a document through the full pipeline.

        Args:
            file_path: Path to document file
            source_filename: Original filename (for metadata). Defaults to basename.

        Returns:
            ProcessedDocument with cleaned text, sections, and metadata

        Raises:
            ValueError: If file validation fails
            Exception: If extraction fails
        """
        if not self.validate_file(file_path):
            raise ValueError(f"File validation failed for {file_path}")

        path = Path(file_path)
        suffix = path.suffix.lower()
        source_file = source_filename or path.name

        try:
            raw_text = ""
            dataframe = None
            parsed_data = None
            sheet_dataframes = None

            if suffix == ".pdf":
                raw_text = self.extract_pdf(file_path)
            elif suffix == ".txt":
                raw_text = self.extract_txt(file_path)
            elif suffix == ".csv":
                dataframe = self.extract_csv(file_path)
                raw_text = dataframe.to_string()
            elif suffix in {".xlsx", ".xls"}:
                sheet_dataframes = self.extract_excel(file_path)
                raw_text = "\n\n".join(
                    f"[Sheet: {name}]\n{df.to_string()}"
                    for name, df in sheet_dataframes.items()
                )
            elif suffix == ".json":
                parsed_data = self.extract_json(file_path)
                raw_text = json_lib.dumps(parsed_data, indent=2, default=str)
            elif suffix in {".yaml", ".yml"}:
                parsed_data = self.extract_yaml(file_path)
                raw_text = str(parsed_data)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            logger.info("Document extraction successful", file_path=file_path)

            # Run full preprocessing pipeline
            processed = preprocess_document(
                raw_text=raw_text,
                file_type=suffix,
                source_file=source_file,
                dataframe=dataframe,
                parsed_data=parsed_data,
                sheet_dataframes=sheet_dataframes,
            )

            logger.info(
                "Document ingestion complete",
                file_path=file_path,
                sections=len(processed.sections),
                cleaned_length=len(processed.cleaned_text),
            )
            return processed

        except Exception as e:
            logger.error("Document ingestion failed", file_path=file_path, error=str(e))
            raise
