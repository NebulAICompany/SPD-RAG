"""Document parsers for the upload pipeline: plain text and Excel (markdown output)."""

import os
import asyncio
import pandas as pd
from backend.shared.logger import get_logger

logger = get_logger("PARSER")


async def TxtParser(file_path: str) -> str:
    """Read a UTF-8 text file and return its contents."""
    try:
        def _read():
            with open(file_path, "r", encoding="utf-8") as infile:
                return infile.read()
        return await asyncio.to_thread(_read)
    except Exception as e:
        logger.error(f"Error parsing text file {file_path}: {e}")
        raise e

async def ExcelParser(file_path: str) -> str:
    """Load all Excel sheets and return a single markdown string (sheets separated)."""
    try:
        df_dict = await asyncio.to_thread(pd.read_excel, file_path, sheet_name=None)
        contents = []
        for sheet_name, sheet_data in df_dict.items():
            sheet_data.fillna("", inplace=True)
            contents.append(
                f"**[Sheet Name:{sheet_name}]**\n\n{sheet_data.to_markdown(index=False)}\n"
            )
        return "\n====SHEET SEPARATOR====\n".join(contents)
    except Exception as e:
        logger.error(f"Error parsing excel file {file_path}: {e}")
        raise e

