"""
Tests for s3_worker sync functionality.

Verifies that sync_to_s3 properly excludes temporary .sql files
from being uploaded to S3 (issue #1794).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import tempfile
import json
from unittest.mock import patch, MagicMock

import pytest

from scripts.s3_worker import sync_to_s3, parse_s3_uri


class TestSyncToS3:
    """Tests for the sync_to_s3 function."""

    def test_exclude_sql_tmp_by_default(self, tmp_path):
        """Default exclude pattern should include *.sql.tmp."""
        result_dir = tmp_path / "results"
        result_dir.mkdir()

        (result_dir / "result.json").write_text('{"test": "data"}')
        (result_dir / "output.sql.tmp").write_text("temp sql content")
        (result_dir / "valid.sql").write_text("valid sql content")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            sync_to_s3(str(result_dir), "s3://test-bucket/results")

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]

            assert "aws" in call_args
            assert "s3" in call_args
            assert "sync" in call_args
            assert "--exclude" in call_args
            exclude_idx = call_args.index("--exclude")
            assert call_args[exclude_idx + 1] == "*.sql.tmp"

    def test_exclude_pattern_in_sync_command(self, tmp_path):
        """Verify *.sql.tmp exclusion is passed to aws CLI."""
        result_dir = tmp_path / "results"
        result_dir.mkdir()
        (result_dir / "test.json").write_text('{}')

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            sync_to_s3(str(result_dir), "s3://my-bucket/campaigns/test/run1")

            call_args = mock_run.call_args[0][0]

            assert "s3://my-bucket/campaigns/test/run1" in call_args

            exclude_indices = [i for i, x in enumerate(call_args) if x == "--exclude"]
            patterns = [call_args[i + 1] for i in exclude_indices]
            assert "*.sql.tmp" in patterns

    def test_multiple_exclude_patterns(self, tmp_path):
        """Can specify multiple exclude patterns."""
        result_dir = tmp_path / "results"
        result_dir.mkdir()
        (result_dir / "test.json").write_text('{}')

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            sync_to_s3(
                str(result_dir),
                "s3://bucket/prefix",
                exclude_patterns=["*.sql.tmp", "*.tmp", "*.log"],
            )

            call_args = mock_run.call_args[0][0]

            exclude_indices = [i for i, x in enumerate(call_args) if x == "--exclude"]
            patterns = [call_args[i + 1] for i in exclude_indices]

            assert "*.sql.tmp" in patterns
            assert "*.tmp" in patterns
            assert "*.log" in patterns

    def test_sync_to_s3_returns_completed_process(self, tmp_path):
        """sync_to_s3 returns a CompletedProcess object."""
        result_dir = tmp_path / "results"
        result_dir.mkdir()
        (result_dir / "test.json").write_text('{}')

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="upload: results/test.json to s3://bucket/prefix/test.json",
                stderr="",
            )

            result = sync_to_s3(str(result_dir), "s3://bucket/prefix")

            assert result.returncode == 0
            assert "test.json" in result.stdout


class TestParseS3Uri:
    """Tests for parse_s3_uri helper."""

    def test_parses_s3_uri_correctly(self):
        """Test S3 URI parsing returns bucket and key."""
        bucket, key = parse_s3_uri("s3://my-bucket/path/to/file.json")
        assert bucket == "my-bucket"
        assert key == "path/to/file.json"

    def test_parses_s3_uri_with_no_key(self):
        """Test S3 URI with just bucket."""
        bucket, key = parse_s3_uri("s3://my-bucket")
        assert bucket == "my-bucket"
        assert key == ""

    def test_invalid_uri_raises_error(self):
        """Test that non-s3 URI raises ValueError."""
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            parse_s3_uri("https://example.com/file.txt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])