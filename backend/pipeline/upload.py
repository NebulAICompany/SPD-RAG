import os
from pathlib import Path
from . import vector as vectorpipe
from ..shared.logger import get_logger
from ..utils.parser import TxtParser, ExcelParser

logger = get_logger("UPLOAD")

async def parse_document(file_path: str):
    """
    Parse document using local parsers.
    """
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == ".txt":
        return await TxtParser(file_path)
    elif file_extension in [".xlsx", ".xls"]:
         return await ExcelParser(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Only .txt and .xlsx are supported in RRM.")

async def process_file(file_path: str) -> dict:
    """
    Process an uploaded file: Parse -> Vectorize -> Index
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Uploaded file not found: {file_path}")

    try:
        # 1. Parse
        extracted_text = await parse_document(file_path)
        
        # 2. Vectorize & Index
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
            "message": f"File processing failed: {str(e)}",
            "file_name": Path(file_path).name,
        }
