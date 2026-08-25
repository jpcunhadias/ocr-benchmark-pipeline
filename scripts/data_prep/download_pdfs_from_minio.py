from scripts.pipeline.core import BUCKET_NAME, LOCAL_PDF_DIR, list_subfolders
from scripts.pipeline.pipeline_loop import select_base_folder, select_source_folder
from src.config import load_env
from src.data.minio_client import connect_to_minio, download_files
from src.utils.logger import get_logger
from src.utils.time_utils import extract_month_from_prefix

logger = get_logger(__name__)


# ----------------------------------------------------------------------
def main():
    load_env("prod")
    minio_client = connect_to_minio()

    # Step 1: Select source (root folder)
    source = select_source_folder(minio_client)

    # Step 2: Select base prefix (e.g. acme/2025-05/)
    base_prefix = select_base_folder(minio_client, root=source)
    logger.info(f"\n▶ Selected base folder: {base_prefix}")

    # Step 3: List document subfolders under base prefix
    folders = list_subfolders(minio_client, base_prefix)
    if not folders:
        logger.info("No subfolders found.")
        return

    logger.info("\nAvailable subfolders:")
    for i, folder in enumerate(folders, 1):
        logger.info(f"{i:2d}. {folder}")

    while True:
        choice = input("\nSelect subfolder number to download PDFs from: ")
        if choice.isdigit() and 1 <= int(choice) <= len(folders):
            selected_folder = folders[int(choice) - 1]
            break
        logger.info("Invalid choice. Please try again.")

    month = extract_month_from_prefix(base_prefix)
    if not month:
        logger.info("Could not extract month from base prefix. Exiting.")
        return

    # Step 4: Construct full prefix and local dir
    prefix_on_minio = f"{base_prefix}{selected_folder}/"
    local_subfolder_dir = LOCAL_PDF_DIR / month / selected_folder
    local_subfolder_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nDownloading PDFs from: {prefix_on_minio}")
    download_files(minio_client, BUCKET_NAME, prefix_on_minio, local_subfolder_dir)
    logger.info(f"\n✓ Download completed to: {local_subfolder_dir}")


if __name__ == "__main__":
    main()
