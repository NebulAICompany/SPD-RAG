import os
import asyncio
import pandas as pd
from backend.shared.logger import get_logger

logger = get_logger("PARSER")

async def TxtParser(file_path: str) -> str:
    """
    Parse a text file and return its content.
    """
    try:
        def _read():
            with open(file_path, "r", encoding="utf-8") as infile:
                return infile.read()
        return await asyncio.to_thread(_read)
    except Exception as e:
        logger.error(f"Error parsing text file {file_path}: {e}")
        raise e

async def ExcelParser(file_path: str) -> str:
    """
    Parse an Excel file and return its content formatted as markdown.
    """
    try:
        df_dict = await asyncio.to_thread(pd.read_excel, file_path, sheet_name=None)
        contents = []
        for sheet_name, sheet_data in df_dict.items():
            # Replace NaN with empty string
            sheet_data.fillna("", inplace=True)
            # Convert to markdown
            contents.append(
                f"**[Sheet Name:{sheet_name}]**\n\n{sheet_data.to_markdown(index=False)}\n"
            )
        return "\n====SHEET SEPARATOR====\n".join(contents)
    except Exception as e:
        logger.error(f"Error parsing excel file {file_path}: {e}")
        raise e

