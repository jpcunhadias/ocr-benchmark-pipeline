import json
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def count_pages_in_json(json_path: Path) -> int:
    """
    Count the number of pages in a JSON file.

    Args:
        json_path (Path): Path to the JSON file.

    Returns:
        int: Number of page entries in the JSON file.
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return len(data.keys())
    except Exception as e:
        logger.error(f"Failed to read {json_path}: {e}")
        return 0


def collect_expected_pages(json_root: Path) -> set[tuple[str, str]]:
    """
    Collect expected (document, page) pairs from OCR JSON outputs.

    Args:
        json_root (Path): Root directory containing OCR JSON files.

    Returns:
        Set of tuples (document, page).
    """
    expected = set()

    for f in json_root.rglob("*.json"):
        subfolder = f.parent.name
        document_name = f.stem
        document = f"{subfolder}/{document_name}"

        try:
            with f.open("r", encoding="utf-8") as j:
                data = json.load(j)
                for page_key in data:
                    expected.add((document, page_key))
        except Exception as e:
            logger.error(f"Failed to load JSON {f}: {e}")

    return expected


def collect_seen_pages(csv_path: Path) -> set[tuple[str, str]]:
    """
    Collect (document, page) pairs that were already processed and saved to CSV.

    Args:
        csv_path (Path): Path to the funcionarios_extraidos.csv file.

    Returns:
        Set of tuples (document, page).
    """
    df = pd.read_csv(csv_path)
    return set(zip(df["document"], df["page"], strict=False))


def main(
    folder: str = "../results/tesseract",
    csv_path: str = "results/funcionarios_extraidos.csv",
) -> None:
    """
    Validate the extracted pages by comparing JSON outputs with CSV records.

    Args:
        folder (str): Directory containing JSON outputs from OCR.
        csv_path (str): Path to the CSV file with extracted data.
    """
    folder_path = Path(folder)
    all_jsons = list(folder_path.rglob("*.json"))

    total_pages = sum(count_pages_in_json(f) for f in all_jsons)
    logger.info(f"Total JSON files found: {len(all_jsons)}")
    logger.info(f"Total pages (keys across all JSONs): {total_pages}")

    expected = collect_expected_pages(folder_path)
    seen = collect_seen_pages(Path(csv_path))

    logger.info(f"Expected (document, page) pairs: {len(expected)}")
    logger.info(f"Seen in CSV: {len(seen)}")

    missing = expected - seen
    logger.info(f"Missing entries: {len(missing)}")
    for doc, page in sorted(missing):
        logger.info(f"- {doc} :: {page}")


if __name__ == "__main__":
    main()
