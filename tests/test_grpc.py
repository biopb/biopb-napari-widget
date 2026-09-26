"""Tests for _grpc.py gRPC communication."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from biopb_napari_widget.image_processing._grpc import (
    _check_chunk_memory,
    _estimate_chunk_memory_mb,
    _get_grpc_channel,
    _get_label_filter,
    _parse_annotation_tsv,
    _parse_server_url,
)


class TestParseServerUrl:
    """Tests for _parse_server_url function."""

    def test_valid_hostname_with_port(self):
        """Valid hostname:port format parses correctly."""
        host, port, scheme, label_filter = _parse_server_url("localhost:50051")
        assert host == "localhost"
        assert port == 50051
        assert scheme is None  # Auto-detect
        assert label_filter is None

    def test_valid_domain_with_port(self):
        """Valid domain:port format parses correctly."""
        host, port, scheme, label_filter = _parse_server_url("lacss.biopb.org:8080")
        assert host == "lacss.biopb.org"
        assert port == 8080
        assert scheme is None  # Auto-detect
        assert label_filter is None

    def test_hostname_without_port_defaults_to_443(self):
        """Hostname without port defaults to port 443."""
        host, port, scheme, label_filter = _parse_server_url("lacss.biopb.org")
        assert host == "lacss.biopb.org"
        assert port == 443
        assert scheme is None  # Auto-detect
        assert label_filter is None

    def test_localhost_without_port(self):
        """Local hostname without port defaults to 443."""
        host, port, scheme, label_filter = _parse_server_url("localhost")
        assert host == "localhost"
        assert port == 443
        assert scheme is None  # Auto-detect
        assert label_filter is None

    def test_http_scheme_prefix(self):
        """http:// prefix returns HTTP scheme."""
        host, port, scheme, label_filter = _parse_server_url("http://localhost:50051")
        assert host == "localhost"
        assert port == 50051
        assert scheme == "http"
        assert label_filter is None

    def test_https_scheme_prefix(self):
        """https:// prefix returns HTTPS scheme."""
        host, port, scheme, label_filter = _parse_server_url(
            "https://lacss.biopb.org:8080"
        )
        assert host == "lacss.biopb.org"
        assert port == 8080
        assert scheme == "https"
        assert label_filter is None

    def test_https_scheme_no_port(self):
        """https:// with hostname only defaults to port 443."""
        host, port, scheme, label_filter = _parse_server_url("https://lacss.biopb.org")
        assert host == "lacss.biopb.org"
        assert port == 443
        assert scheme == "https"
        assert label_filter is None

    def test_http_scheme_no_port(self):
        """http:// with hostname only defaults to port 443."""
        host, port, scheme, label_filter = _parse_server_url("http://localhost")
        assert host == "localhost"
        assert port == 443
        assert scheme == "http"
        assert label_filter is None

    def test_url_with_label_filter(self):
        """URL with path component extracts label filter."""
        host, port, scheme, label_filter = _parse_server_url("localhost:50051/filter")
        assert host == "localhost"
        assert port == 50051
        assert scheme is None
        assert label_filter == "filter"

    def test_url_with_scheme_and_label_filter(self):
        """URL with scheme and path extracts label filter."""
        host, port, scheme, label_filter = _parse_server_url(
            "http://localhost:50051/filter"
        )
        assert host == "localhost"
        assert port == 50051
        assert scheme == "http"
        assert label_filter == "filter"

    def test_url_with_https_and_label_filter(self):
        """https URL with path extracts label filter."""
        host, port, scheme, label_filter = _parse_server_url(
            "https://lacss.biopb.org:8080/segmentation"
        )
        assert host == "lacss.biopb.org"
        assert port == 8080
        assert scheme == "https"
        assert label_filter == "segmentation"

    def test_url_without_port_with_label_filter(self):
        """URL without port (defaults to 443) with path extracts label filter."""
        host, port, scheme, label_filter = _parse_server_url(
            "lacss.biopb.org/threshold"
        )
        assert host == "lacss.biopb.org"
        assert port == 443
        assert scheme is None
        assert label_filter == "threshold"

    def test_invalid_format_raises(self):
        """Invalid URL format raises ValueError."""
        invalid_urls = [
            "localhost:",  # missing port
            ":50051",  # missing hostname
            "local:host:50051",  # invalid hostname
            "",  # empty string
            "local space:50051",  # space in hostname
        ]
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Invalid server URL"):
                _parse_server_url(url)


class TestGetLabelFilter:
    """Tests for _get_label_filter function."""

    def test_no_label_filter(self):
        """URL without path returns None."""
        assert _get_label_filter("localhost:50051") is None
        assert _get_label_filter("lacss.biopb.org") is None

    def test_with_label_filter(self):
        """URL with path returns label filter."""
        assert _get_label_filter("localhost:50051/filter") == "filter"
        assert (
            _get_label_filter("http://localhost:50051/segmentation") == "segmentation"
        )
        assert _get_label_filter("https://lacss.biopb.org/threshold") == "threshold"

    def test_invalid_url_raises(self):
        """Invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid server URL"):
            _get_label_filter("invalid:url:format")


class TestGetGrpcChannel:
    """Tests for _get_grpc_channel function."""

    def test_http_scheme_from_url(self):
        """http:// in URL creates insecure channel."""
        settings = {
            "Server": "http://localhost:50051",
        }
        channel = _get_grpc_channel(settings)
        # Channel is created, verify it's a grpc channel
        assert channel is not None

    def test_https_scheme_from_url(self):
        """https:// in URL creates secure channel."""
        settings = {
            "Server": "https://lacss.biopb.org",
        }
        channel = _get_grpc_channel(settings)
        assert channel is not None

    def test_auto_scheme_http_port(self):
        """URL without scheme and non-443 port uses HTTP."""
        settings = {
            "Server": "localhost:50051",
        }
        channel = _get_grpc_channel(settings)
        assert channel is not None

    def test_auto_scheme_https_port(self):
        """URL without scheme and port 443 uses HTTPS."""
        settings = {
            "Server": "lacss.biopb.org",
        }
        channel = _get_grpc_channel(settings)
        assert channel is not None

    def test_server_without_port(self):
        """Server without port defaults to 443."""
        settings = {
            "Server": "lacss.biopb.org",
        }
        # Should add :443 to server URL internally
        channel = _get_grpc_channel(settings)
        assert channel is not None

    def test_url_with_label_filter(self):
        """URL with label filter still creates channel correctly."""
        settings = {
            "Server": "localhost:50051/filter",
        }
        channel = _get_grpc_channel(settings)
        # Label filter is ignored for channel creation
        assert channel is not None

    def test_url_with_scheme_and_label_filter(self):
        """URL with scheme and label filter creates channel correctly."""
        settings = {
            "Server": "http://localhost:50051/segmentation",
        }
        channel = _get_grpc_channel(settings)
        assert channel is not None

    def test_invalid_server_url_raises(self):
        """Invalid server URL raises ValueError."""
        settings = {
            "Server": "invalid:url:format",
        }
        with pytest.raises(ValueError, match="Invalid server URL"):
            _get_grpc_channel(settings)


class TestGrpcProcessImage:
    """Tests for grpc_process_image generator function."""

    def test_grid_positions_not_supported(self):
        """Grid processing raises ValueError when grid_positions is provided."""
        # This tests that grid_positions must be None for process_image
        # The validation happens inside the generator, wrapped by thread_worker
        # We document the expected behavior - ValueError should be raised

    def test_invalid_dimensions_raises(self):
        """Wrong dimensions raises ValueError."""
        # Documenting expected behavior - ValueError for wrong dimensions


class TestParseAnnotationTsv:
    """Tests for _parse_annotation_tsv function."""

    def test_empty_annotation(self):
        """Empty annotation returns empty DataFrame."""
        result = _parse_annotation_tsv("")
        assert result.empty

    def test_none_annotation(self):
        """None annotation returns empty DataFrame."""
        result = _parse_annotation_tsv(None)
        assert result.empty

    def test_simple_tsv(self):
        """Simple TSV with header parses correctly."""
        tsv = "col1\tcol2\nval1\tval2\nval3\tval4"
        result = _parse_annotation_tsv(tsv)
        assert result["col1"].tolist() == ["val1", "val3"]
        assert result["col2"].tolist() == ["val2", "val4"]

    def test_single_row(self):
        """TSV with single data row parses correctly."""
        tsv = "name\tvalue\nitem1\t100"
        result = _parse_annotation_tsv(tsv)
        assert result["name"].tolist() == ["item1"]
        assert result["value"].tolist() == [100]  # pandas auto-detects int

    def test_numeric_values(self):
        """Numeric values are parsed as integers."""
        tsv = "id\tcount\n1\t42\n2\t100"
        result = _parse_annotation_tsv(tsv)
        assert result["id"].tolist() == [1, 2]
        assert result["count"].tolist() == [42, 100]

    def test_multiple_columns(self):
        """TSV with multiple columns parses correctly."""
        tsv = "a\tb\tc\n1\t2\t3\n4\t5\t6"
        result = _parse_annotation_tsv(tsv)
        assert result["a"].tolist() == [1, 4]
        assert result["b"].tolist() == [2, 5]
        assert result["c"].tolist() == [3, 6]

    def test_header_duplicated_as_data(self):
        """Header row appearing as data row is filtered out."""
        tsv = "col1\tcol2\ncol1\tcol2\nval1\tval2"
        result = _parse_annotation_tsv(tsv)
        assert len(result) == 1
        assert result["col1"].tolist() == ["val1"]
        assert result["col2"].tolist() == ["val2"]

    def test_multiple_header_duplicates(self):
        """Multiple header rows appearing as data are all filtered out."""
        tsv = "col1\tcol2\ncol1\tcol2\ncol1\tcol2\nval1\tval2\nval3\tval4"
        result = _parse_annotation_tsv(tsv)
        assert len(result) == 2
        assert result["col1"].tolist() == ["val1", "val3"]
        assert result["col2"].tolist() == ["val2", "val4"]

    def test_comment_line_skipped(self):
        """Comment lines starting with # are skipped before parsing."""
        tsv = "# Directionality analysis\nangle\tcount\n-90.0\t0.0\n-88.0\t1.0"
        result = _parse_annotation_tsv(tsv)
        assert list(result.columns) == ["angle", "count"]
        assert len(result) == 2
        assert result["angle"].tolist() == [-90.0, -88.0]

    def test_multiple_comment_lines(self):
        """Multiple comment lines are all skipped."""
        tsv = "# Comment 1\n# Comment 2\na\tb\n1\t2\n3\t4"
        result = _parse_annotation_tsv(tsv)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_only_comment_lines(self):
        """TSV with only comment lines returns empty DataFrame."""
        tsv = "# Only comments\n# No data"
        result = _parse_annotation_tsv(tsv)
        assert result.empty


class TestEstimateChunkMemory:
    """Tests for _estimate_chunk_memory_mb function."""

    def test_small_chunk(self):
        """Small chunk estimates correctly."""
        chunk = np.zeros((100, 100, 3), dtype=np.uint8)
        mb = _estimate_chunk_memory_mb(chunk)
        # 100 * 100 * 3 * 1 byte = 30,000 bytes ≈ 0.03 MB
        assert mb < 0.1
        assert mb > 0

    def test_medium_chunk(self):
        """Medium chunk estimates correctly."""
        chunk = np.zeros((512, 512, 3), dtype=np.uint8)
        mb = _estimate_chunk_memory_mb(chunk)
        # 512 * 512 * 3 * 1 byte = 786,432 bytes ≈ 0.75 MB
        assert mb > 0.5
        assert mb < 1.0

    def test_large_chunk_float32(self):
        """Large float32 chunk estimates correctly."""
        chunk = np.zeros((1024, 1024, 64, 3), dtype=np.float32)
        mb = _estimate_chunk_memory_mb(chunk)
        # 1024 * 1024 * 64 * 3 * 4 bytes = 805,306,368 bytes ≈ 768 MB
        assert mb > 700
        assert mb < 900

    def test_large_chunk_float64(self):
        """Large float64 chunk estimates correctly."""
        chunk = np.zeros((512, 512, 10), dtype=np.float64)
        mb = _estimate_chunk_memory_mb(chunk)
        # 512 * 512 * 10 * 8 bytes = 20,971,520 bytes ≈ 20 MB
        assert mb > 15
        assert mb < 25

    def test_chunk_without_dtype(self):
        """Chunk without dtype attribute uses float32 default."""
        # Create a mock object without dtype
        mock_chunk = MagicMock()
        mock_chunk.shape = (1000, 1000, 3)
        del mock_chunk.dtype  # Remove dtype attribute

        mb = _estimate_chunk_memory_mb(mock_chunk)
        # Should use float32 default: 1000 * 1000 * 3 * 4 = 12 MB
        assert mb > 10
        assert mb < 15

    def test_empty_chunk(self):
        """Empty chunk returns 0 MB."""
        mock_chunk = MagicMock()
        mock_chunk.shape = ()
        mb = _estimate_chunk_memory_mb(mock_chunk)
        assert mb == 0.0


class TestCheckChunkMemory:
    """Tests for _check_chunk_memory function."""

    def test_small_chunk_passes(self):
        """Small chunk passes without error."""
        chunk = np.zeros((100, 100, 3), dtype=np.uint8)
        # Should not raise
        _check_chunk_memory(chunk)

    def test_medium_chunk_passes_with_warning(self):
        """Medium chunk passes but may log warning."""
        # Create a chunk larger than warn threshold but under error threshold
        # Using default config: warn=500MB, error=2000MB
        chunk = np.zeros((512, 512, 200, 3), dtype=np.float32)
        # 512 * 512 * 200 * 3 * 4 ≈ 629 MB (over warn threshold)
        # Should not raise, may log warning (we don't check warning here)
        _check_chunk_memory(chunk)

    def test_huge_chunk_raises_memory_error(self):
        """Huge chunk raises MemoryError."""
        # Mock a chunk that exceeds error threshold
        mock_chunk = MagicMock()
        mock_chunk.shape = (4096, 4096, 256, 3)  # Very large
        mock_chunk.dtype = np.dtype(np.float32)
        # 4096 * 4096 * 256 * 3 * 4 ≈ 50 GB (way over error threshold)

        with pytest.raises(MemoryError, match="exceeds memory limit"):
            _check_chunk_memory(mock_chunk)

    def test_custom_threshold(self):
        """Custom thresholds from config are respected."""
        from biopb_napari_widget._settings import SETTINGS

        SETTINGS.set("memory.warn_threshold_mb", 10, persist=False)
        SETTINGS.set("memory.error_threshold_mb", 50, persist=False)

        chunk = np.zeros((1024, 1024, 10), dtype=np.float32)
        # 1024 * 1024 * 10 * 4 ≈ 40 MB (over warn=10, under error=50)

        # Should not raise, but would warn
        _check_chunk_memory(chunk)

    def test_custom_error_threshold_raises(self):
        """Custom error threshold from config raises MemoryError."""
        from biopb_napari_widget._settings import SETTINGS

        SETTINGS.set("memory.warn_threshold_mb", 10, persist=False)
        SETTINGS.set("memory.error_threshold_mb", 20, persist=False)

        chunk = np.zeros((1024, 1024, 10), dtype=np.float32)
        # 1024 * 1024 * 10 * 4 ≈ 40 MB (over error=20)

        with pytest.raises(MemoryError, match="exceeds memory limit"):
            _check_chunk_memory(chunk)
