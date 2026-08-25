import csv
import time
from datetime import datetime
from pathlib import Path

import psutil

RESOURCE_CSV_PATH = Path("results/resource_report.csv")


def append_dict_row(row: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def monitor_pipeline_resources(
    folder_name: str, engine_name: str, start_time: float
) -> None:
    """
    Monitors and saves resource usage after OCR execution.

    Args:
        folder_name (str): Name of the processed folder (e.g., 'acme_2025-04')
        engine_name (str): Name of the OCR engine used (e.g., 'tesseract')
        start_time (float): Start timestamp (use time.time())
    """
    elapsed = time.time() - start_time

    # Capture system memory and CPU usage at the final moment
    ram_used_mb = psutil.virtual_memory().used / 1024**2
    cpu_percent = psutil.cpu_percent(interval=1.0)  # Global mean in 1 second

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "engine": engine_name,
        "folder": folder_name,
        "elapsed_sec": round(elapsed, 2),
        "ram_used_mb": round(ram_used_mb, 2),
        "cpu_percent": round(cpu_percent, 2),
    }

    append_dict_row(row, RESOURCE_CSV_PATH)
