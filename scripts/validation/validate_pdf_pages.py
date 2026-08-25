import csv
from datetime import datetime
from pathlib import Path

import pytesseract

from src.data.preprocess import preprocess_image
from src.utils.logger import get_logger

logger = get_logger(__name__)
VALIDATION_REPORT_PATH = Path("results/validation_report.csv")


def validate_folder(img_dir: Path, folder_name: str, month: str = None) -> None:
    results = []
    image_files = sorted(img_dir.rglob("*.png"))

    if not image_files:
        logger.info(f"[{folder_name}] No images found in {img_dir}.")
        return

    for img_path in image_files:
        page_num = img_path.stem.split("_")[-1]
        try:
            processed_img = preprocess_image(str(img_path))
            text = pytesseract.image_to_string(processed_img, lang="eng")
            n_chars = len(text.strip())

            status = "ok" if n_chars > 20 else "empty_ocr"  # adjustable threshold

            results.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "month": month if month else "",
                    "folder": folder_name,
                    "file": img_path.name,
                    "page": page_num,
                    "status": status,
                    "n_chars": n_chars,
                    "error": "",
                }
            )

        except Exception as e:
            results.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "folder": folder_name,
                    "file": img_path.name,
                    "page": page_num,
                    "status": "error",
                    "n_chars": 0,
                    "error": str(e).replace("\n", " ")[:200],  # sanitize line breaks
                }
            )

    # Save incrementally to the central report
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = VALIDATION_REPORT_PATH.exists()

    with open(VALIDATION_REPORT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    logger.info(
        f"[{folder_name}] {len(results)} pages validated. Report saved to: {VALIDATION_REPORT_PATH}"
    )


# ───────────────────────── CLI USAGE ─────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate image pages with a quick OCR pass"
    )
    parser.add_argument(
        "--img_dir", type=str, required=True, help="Directory with PNG images"
    )
    parser.add_argument(
        "--folder", type=str, required=True, help="Subfolder name"
    )

    args = parser.parse_args()
    validate_folder(Path(args.img_dir), args.folder)
