import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.api.services.page_image import (
    MinioConnectionError,
    _find_source_object,
    _render_page_sync,
    render_page_image,
)


class _FakeObj:
    def __init__(self, object_name: str):
        self.object_name = object_name


class _FakeMinioClient:
    def __init__(self, objects_by_prefix: dict[str, list[str]]):
        self._objects_by_prefix = objects_by_prefix

    def list_objects(self, bucket, prefix, recursive=True):
        return [_FakeObj(name) for name in self._objects_by_prefix.get(prefix, [])]


def test_find_source_object_matches_by_stem():
    client = _FakeMinioClient(
        {
            "acme/2025-04/sample-report/": [
                "acme/2025-04/sample-report/sample_delivery_report.pdf"
            ]
        }
    )
    result = _find_source_object(
        client, "bucket", ["acme/2025-04/sample-report/"], "sample_delivery_report"
    )
    assert result == "acme/2025-04/sample-report/sample_delivery_report.pdf"


def test_find_source_object_checks_all_candidate_prefixes():
    client = _FakeMinioClient(
        {
            "acme/2025-04/other/": ["acme/2025-04/other/unrelated.pdf"],
            "acme/2025-04/sample-report/": [
                "acme/2025-04/sample-report/sample_delivery_report.pdf"
            ],
        }
    )
    result = _find_source_object(
        client,
        "bucket",
        ["acme/2025-04/other/", "acme/2025-04/sample-report/"],
        "sample_delivery_report",
    )
    assert result == "acme/2025-04/sample-report/sample_delivery_report.pdf"


def test_find_source_object_returns_none_when_no_match():
    client = _FakeMinioClient({"acme/2025-04/sample-report/": ["irrelevant.pdf"]})
    result = _find_source_object(
        client, "bucket", ["acme/2025-04/sample-report/"], "sample_delivery_report"
    )
    assert result is None


def test_find_source_object_propagates_backend_errors():
    class _RaisingClient:
        def list_objects(self, bucket, prefix, recursive=True):
            raise RuntimeError("connection refused")

    with pytest.raises(RuntimeError):
        _find_source_object(_RaisingClient(), "bucket", ["prefix/"], "doc")


@patch("src.api.services.page_image._connect_to_minio")
@patch("src.api.services.page_image._find_source_object")
@patch("src.api.services.page_image.convert_from_bytes")
def test_render_page_sync_returns_none_when_no_pages_rendered(
    mock_convert, mock_find, mock_connect
):
    mock_find.return_value = "acme/2025-04/sample-report/sample_delivery_report.pdf"
    mock_connect.return_value = MagicMock()
    mock_convert.return_value = []  # stale/mismatched page number

    result = _render_page_sync(
        "sample_delivery_report", 99, ["acme/2025-04/sample-report/"]
    )

    assert result is None


@patch("src.api.services.page_image._connect_to_minio")
@patch("src.api.services.page_image._find_source_object")
def test_render_page_sync_returns_none_when_no_object_found(mock_find, mock_connect):
    mock_find.return_value = None
    mock_connect.return_value = MagicMock()

    result = _render_page_sync(
        "sample_delivery_report", 1, ["acme/2025-04/sample-report/"]
    )

    assert result is None


@patch("src.api.services.page_image._connect_to_minio")
@patch("src.api.services.page_image._find_source_object")
@patch("src.api.services.page_image.convert_from_bytes")
def test_render_page_sync_returns_png_bytes_on_success(
    mock_convert, mock_find, mock_connect
):
    mock_find.return_value = "acme/2025-04/sample-report/sample_delivery_report.pdf"
    fake_response = MagicMock()
    fake_response.read.return_value = b"%PDF-fake-bytes"
    fake_client = MagicMock()
    fake_client.get_object.return_value = fake_response
    mock_connect.return_value = fake_client
    mock_convert.return_value = [Image.new("RGB", (10, 10))]

    result = _render_page_sync(
        "sample_delivery_report", 1, ["acme/2025-04/sample-report/"]
    )

    assert result is not None
    assert result.startswith(b"\x89PNG")
    fake_response.close.assert_called_once()
    fake_response.release_conn.assert_called_once()


@patch("src.api.services.page_image._connect_to_minio")
@patch("src.api.services.page_image._find_source_object")
def test_render_page_sync_wraps_s3_errors(mock_find, mock_connect):
    from minio.error import S3Error

    mock_find.side_effect = S3Error(
        "InternalError", "boom", "resource", "req-id", "host-id", response=None
    )
    mock_connect.return_value = MagicMock()

    with pytest.raises(MinioConnectionError):
        _render_page_sync("sample_delivery_report", 1, ["acme/2025-04/sample-report/"])


@patch("src.api.services.page_image._connect_to_minio")
@patch("src.api.services.page_image._find_source_object")
def test_render_page_sync_wraps_connection_errors(mock_find, mock_connect):
    """Regression test: a refused/unresolvable connection (MinIO not
    running) surfaces as urllib3.exceptions.HTTPError (typically
    MaxRetryError), NOT minio.error.S3Error -- confirmed live by stopping
    the MinIO container and hitting the real endpoint, which returned an
    unhandled 500 before this was caught."""
    import urllib3.exceptions

    mock_find.side_effect = urllib3.exceptions.HTTPError("connection refused")
    mock_connect.return_value = MagicMock()

    with pytest.raises(MinioConnectionError):
        _render_page_sync("sample_delivery_report", 1, ["acme/2025-04/sample-report/"])


@patch(
    "src.api.services.page_image._fetch_direct_source_object", new_callable=AsyncMock
)
@patch("src.api.services.page_image._render_direct_sync")
def test_render_page_image_prefers_direct_source_object(
    mock_render_direct, mock_fetch_direct
):
    """When ocr_page_metrics.source_object is set (pipeline recorded it at
    write time), render_page_image must use it directly and never touch
    documents/MinIO listing at all."""
    mock_fetch_direct.return_value = (
        "acme/2025-04/sample-report/sample_delivery_report.pdf"
    )
    mock_render_direct.return_value = b"fake-png-bytes"

    with patch(
        "src.api.services.page_image._fetch_candidate_source_paths",
        new_callable=AsyncMock,
    ) as mock_fallback:
        result = asyncio.run(render_page_image("run-1", "sample_delivery_report", 1))

        assert result == b"fake-png-bytes"
        mock_render_direct.assert_called_once_with(
            "acme/2025-04/sample-report/sample_delivery_report.pdf", 1
        )
        mock_fallback.assert_not_called()


@patch(
    "src.api.services.page_image._fetch_direct_source_object", new_callable=AsyncMock
)
@patch(
    "src.api.services.page_image._fetch_candidate_source_paths", new_callable=AsyncMock
)
@patch("src.api.services.page_image._render_page_sync")
def test_render_page_image_falls_back_when_no_source_object(
    mock_render_fallback, mock_fetch_candidates, mock_fetch_direct
):
    """Runs that predate the source_object column must still work, via the
    original prefix + filename-stem search."""
    mock_fetch_direct.return_value = None
    mock_fetch_candidates.return_value = ["acme/2025-04/sample-report/"]
    mock_render_fallback.return_value = b"fake-png-bytes"

    result = asyncio.run(render_page_image("run-1", "sample_delivery_report", 1))

    assert result == b"fake-png-bytes"
    mock_render_fallback.assert_called_once_with(
        "sample_delivery_report", 1, ["acme/2025-04/sample-report/"]
    )


@patch(
    "src.api.services.page_image._fetch_direct_source_object", new_callable=AsyncMock
)
@patch(
    "src.api.services.page_image._fetch_candidate_source_paths", new_callable=AsyncMock
)
def test_render_page_image_returns_none_with_no_candidates(
    mock_fetch_candidates, mock_fetch_direct
):
    mock_fetch_direct.return_value = None
    mock_fetch_candidates.return_value = []

    result = asyncio.run(render_page_image("run-1", "sample_delivery_report", 1))

    assert result is None
