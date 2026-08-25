import os

from src.utils import runtime


def setup_function(_):
    runtime.reset_runtime_caches()


def teardown_function(_):
    runtime.reset_runtime_caches()
    for key in ("PIPELINE_PUBLISH_ENABLED", "PIPELINE_USE_MINIO"):
        if key in os.environ:
            del os.environ[key]


def test_pipeline_publish_enabled_defaults_true():
    if "PIPELINE_PUBLISH_ENABLED" in os.environ:
        del os.environ["PIPELINE_PUBLISH_ENABLED"]
    runtime.reset_runtime_caches()
    assert runtime.pipeline_publish_enabled() is True


def test_pipeline_publish_enabled_false_via_env():
    os.environ["PIPELINE_PUBLISH_ENABLED"] = "false"
    runtime.reset_runtime_caches()
    assert runtime.pipeline_publish_enabled() is False


def test_pipeline_use_minio_defaults_true():
    if "PIPELINE_USE_MINIO" in os.environ:
        del os.environ["PIPELINE_USE_MINIO"]
    runtime.reset_runtime_caches()
    assert runtime.pipeline_use_minio() is True


def test_pipeline_use_minio_disabled_via_env():
    os.environ["PIPELINE_USE_MINIO"] = "0"
    runtime.reset_runtime_caches()
    assert runtime.pipeline_use_minio() is False
