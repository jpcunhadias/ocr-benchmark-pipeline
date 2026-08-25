import re


def extract_month_from_prefix(prefix: str) -> str:
    """
    Extracts 'YYYY-MM' from a folder path like 'ocr/2025-04'.
    """
    match = re.search(r"(20\d{2})[-_/]?(\d{2})", prefix)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    raise ValueError(f"[ERROR] Could not extract month from prefix: {prefix}")
