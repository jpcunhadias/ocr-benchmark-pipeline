import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from src.config import load_env

# Ensure logging is configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_environment(env_path: str | Path | None = None) -> None:
    """
    Load environment variables from a .env file.

    Parameters
    ----------
    env_path : Optional[Union[str, Path]]
        Path to the .env file. Defaults to ".env.prod".
    """

    load_env(env="prod" if env_path is None else str(env_path))
    logger.info("Environment variables loaded from %s", env_path)
    logger.debug("MINIO_ENDPOINT: %s", os.getenv("MINIO_ENDPOINT"))
    logger.debug("MINIO_ACCESS_KEY: %s", os.getenv("MINIO_ACCESS_KEY"))
    logger.debug("MINIO_SECRET_KEY: %s", os.getenv("MINIO_SECRET_KEY"))


def _sanitize_endpoint(raw: str) -> tuple[str, bool]:
    """
    Accepts 'minio:9000', 'http://minio:9000', 'http://minio:9000/' (no path),
    or even accidentally 'http://minio:9000/some/path' and strips the path.
    Returns (host:port, secure).
    """
    if not raw:
        return "minio:9000", False

    # If plain host:port, no scheme
    if "://" not in raw:
        return raw.rstrip("/"), False

    u = urlparse(raw)
    if u.path and u.path != "/":
        # path is not allowed by the SDK; drop it
        logger.warning("MINIO_ENDPOINT had a path '%s' — stripping it.", u.path)

    host = u.hostname or "minio"
    port = u.port or (443 if u.scheme == "https" else 80)
    secure = u.scheme == "https"
    return f"{host}:{port}", secure


def connect_to_minio() -> Minio:
    """
    Establishes a connection to a MinIO server using environment variables.
    Retrieves the MinIO endpoint, access key, and secret key from environment variables,
    with default values provided if the variables are not set. The endpoint is sanitized
    to determine whether the connection should be secure (HTTPS) or not (HTTP).
    Logs the connection attempt and returns an initialized Minio client.
    Returns
    -------
    Minio
        An instance of the Minio client connected to the specified server.
    Notes
    -----
    Environment variables used:
        - MINIO_ENDPOINT: The URL of the MinIO server (default: "http://minio:9000").
        - MINIO_ACCESS_KEY: The access key for authentication (default: "minio").
        - MINIO_SECRET_KEY: The secret key for authentication (default: "minio123").
    """

    raw_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minio")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minio123")

    endpoint, secure = _sanitize_endpoint(raw_endpoint)
    logger.info("Connecting to MinIO server at %s (secure=%s)", endpoint, secure)

    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def list_files(
    minio_client: Minio,
    bucket_name: str,
    prefix: str = "",
) -> list[str]:
    """
    List all object names in a bucket (optionally under a prefix).

    Parameters
    ----------
    minio_client : Minio
        Connected Minio client.
    bucket_name : str
        Name of the bucket to list.
    prefix : str, optional
        Restrict listing to objects that start with this prefix.

    Returns
    -------
    List[str]
        Object names found (empty list on error).
    """
    try:
        logger.info(
            "Listing files in bucket '%s' with prefix '%s'", bucket_name, prefix
        )
        return [
            obj.object_name
            for obj in minio_client.list_objects(
                bucket_name,
                prefix=prefix,
                recursive=True,
            )
            if obj.object_name is not None
        ]
    except S3Error as err:
        logger.error(f"[MinIO] Listing error: {err}")
        return []


def download_files(
    minio_client: Minio,
    bucket_name: str,
    prefix: str,
    local_dir: str | Path,
) -> None:
    """
    Download every object under `prefix` from `bucket_name` into `local_dir`.

    Existing files are overwritten.

    Parameters
    ----------
    minio_client : Minio
        Connected Minio client.
    bucket_name : str
        Bucket to download from.
    prefix : str
        Prefix (folder path) inside the bucket.
    local_dir : Union[str, Path]
        Local destination directory.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    objects = list_files(minio_client, bucket_name, prefix)
    for obj_name in objects:
        destination = local_dir / Path(obj_name).name
        logger.info(f"Downloading {obj_name} -> {destination}")
        try:
            minio_client.fget_object(bucket_name, obj_name, str(destination))
        except S3Error as err:
            logger.error(f"Download failed for {obj_name}: {err}")


# Ensure the environment is loaded at the start of the script
load_environment("prod")
