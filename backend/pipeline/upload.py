"""Upload pipeline: parse supported documents and index them for SPD-RAG retrieval.

Supported formats: .txt, .xlsx, .xls. Parsed text is chunked and embedded via
VectorStorePipeline and stored in the Qdrant 'documents' collection.
"""

import os
from pathlib import Path
from . import vector as vectorpipe
from ..shared.logger import get_logger
from ..utils.parser import TxtParser, ExcelParser

logger = get_logger("UPLOAD")


async def parse_document(file_path: str) -> str:
    """Parse a document based on its extension; returns raw text (markdown for Excel)."""
    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".txt":
        return await TxtParser(file_path)
    if file_extension in [".xlsx", ".xls"]:
        return await ExcelParser(file_path)
    raise ValueError(
        f"Unsupported file type: {file_extension}. Only .txt and .xlsx/.xls are supported."
    )


async def process_file(file_path: str) -> dict:
    """Parse the file, then chunk, embed, and index it in the vectorstore."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Uploaded file not found: {file_path}")

    try:
        extracted_text = await parse_document(file_path)
        original_stem = Path(file_path).stem

        pipeline = vectorpipe.VectorStorePipeline()
        await pipeline.run(
            text_content=extracted_text,
            document_name=original_stem,
        )

        return {
            "status": "success",
            "message": "File processed successfully",
            "file_name": original_stem,
            "text_length": len(extracted_text),
        }

    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        return {
            "status": "error",
            "message": f"File processing failed: {e}",
            "file_name": Path(file_path).name,
        }
