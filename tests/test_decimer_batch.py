"""
tests/test_decimer_batch.py
---------------------------
Unit and integration tests for decimer_batch.py.

Run with:
    pytest tests/test_decimer_batch.py -v
    pytest tests/test_decimer_batch.py -v --log-cli-level=DEBUG

All tests mock predict_SMILES so the DECIMER model is never loaded.
"""

import csv
import logging
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers: import the module under test without triggering model load
# ---------------------------------------------------------------------------

# DECIMER is imported lazily inside main(); we patch it at that point.
import importlib
import types

# Insert the repo root on sys.path so we can import decimer_batch directly.
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import decimer_batch as db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_image_dir(tmp_path):
    """A temporary directory pre-populated with valid dummy PNG images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    return img_dir


def make_png(path: Path, size: int = 4096) -> Path:
    """Write a file with a valid PNG magic header followed by padding."""
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (size - 8))
    return path


def make_jpeg(path: Path, size: int = 4096) -> Path:
    """Write a file with a valid JPEG magic header followed by padding."""
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * (size - 4))
    return path


def make_small_file(path: Path) -> Path:
    """Write a file smaller than the default minimum."""
    path.write_bytes(b"\x89PNG" + b"\x00" * 10)  # 14 bytes < 1024 minimum
    return path


def make_bad_magic_png(path: Path) -> Path:
    """PNG extension but JPEG magic bytes — should fail validation."""
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 4092)
    return path


def make_bad_magic_jpeg(path: Path) -> Path:
    """JPEG extension but PNG magic bytes — should fail validation."""
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4088)
    return path


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:

    def test_returns_named_logger(self):
        log = db.setup_logging(None, verbose=False)
        assert log.name == "decimer_batch"

    def test_propagate_is_false(self):
        log = db.setup_logging(None, verbose=False)
        assert log.propagate is False

    def test_single_console_handler_when_no_log_file(self):
        log = db.setup_logging(None, verbose=False)
        assert len(log.handlers) == 1
        assert isinstance(log.handlers[0], logging.StreamHandler)

    def test_two_handlers_when_log_file_given(self, tmp_path):
        log_file = tmp_path / "test.log"
        log = db.setup_logging(log_file, verbose=False)
        assert len(log.handlers) == 2
        handler_types = {type(h) for h in log.handlers}
        assert logging.handlers.RotatingFileHandler in handler_types

    def test_console_level_info_by_default(self):
        log = db.setup_logging(None, verbose=False)
        console = log.handlers[0]
        assert console.level == logging.INFO

    def test_console_level_debug_when_verbose(self):
        log = db.setup_logging(None, verbose=True)
        console = log.handlers[0]
        assert console.level == logging.DEBUG

    def test_file_handler_always_debug(self, tmp_path):
        log_file = tmp_path / "test.log"
        log = db.setup_logging(log_file, verbose=False)
        file_handler = next(
            h for h in log.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        )
        assert file_handler.level == logging.DEBUG

    def test_log_file_created(self, tmp_path):
        log_file = tmp_path / "subdir" / "test.log"
        log = db.setup_logging(log_file, verbose=False)
        log.info("test message")
        assert log_file.exists()

    def test_no_duplicate_handlers_on_repeated_calls(self):
        db.setup_logging(None, verbose=False)
        log = db.setup_logging(None, verbose=False)
        assert len(log.handlers) == 1

    def test_rotating_file_max_bytes(self, tmp_path):
        log_file = tmp_path / "test.log"
        log = db.setup_logging(log_file, verbose=False)
        file_handler = next(
            h for h in log.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        )
        assert file_handler.maxBytes == 5 * 1024 * 1024

    def test_rotating_file_backup_count(self, tmp_path):
        log_file = tmp_path / "test.log"
        log = db.setup_logging(log_file, verbose=False)
        file_handler = next(
            h for h in log.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        )
        assert file_handler.backupCount == 3


# ---------------------------------------------------------------------------
# collect_images
# ---------------------------------------------------------------------------

class TestCollectImages:

    def test_finds_png(self, tmp_image_dir):
        make_png(tmp_image_dir / "mol.png")
        images = db.collect_images(tmp_image_dir, {".png"}, recursive=False)
        assert len(images) == 1
        assert images[0].name == "mol.png"

    def test_finds_jpeg(self, tmp_image_dir):
        make_jpeg(tmp_image_dir / "mol.jpg")
        images = db.collect_images(tmp_image_dir, {".jpg"}, recursive=False)
        assert len(images) == 1

    def test_ignores_wrong_extension(self, tmp_image_dir):
        (tmp_image_dir / "mol.txt").write_text("not an image")
        images = db.collect_images(tmp_image_dir, {".png"}, recursive=False)
        assert images == []

    def test_returns_empty_for_empty_dir(self, tmp_image_dir):
        images = db.collect_images(tmp_image_dir, {".png"}, recursive=False)
        assert images == []

    def test_sorted_alphabetically(self, tmp_image_dir):
        for name in ("c.png", "a.png", "b.png"):
            make_png(tmp_image_dir / name)
        images = db.collect_images(tmp_image_dir, {".png"}, recursive=False)
        assert [i.name for i in images] == ["a.png", "b.png", "c.png"]

    def test_non_recursive_ignores_subdirectory(self, tmp_image_dir):
        subdir = tmp_image_dir / "sub"
        subdir.mkdir()
        make_png(subdir / "nested.png")
        images = db.collect_images(tmp_image_dir, {".png"}, recursive=False)
        assert images == []

    def test_recursive_finds_nested_files(self, tmp_image_dir):
        subdir = tmp_image_dir / "sub"
        subdir.mkdir()
        make_png(subdir / "nested.png")
        images = db.collect_images(tmp_image_dir, {".png"}, recursive=True)
        assert len(images) == 1
        assert images[0].name == "nested.png"

    def test_multiple_extensions(self, tmp_image_dir):
        make_png(tmp_image_dir / "a.png")
        make_jpeg(tmp_image_dir / "b.jpg")
        (tmp_image_dir / "c.txt").write_text("skip me")
        images = db.collect_images(tmp_image_dir, {".png", ".jpg"}, recursive=False)
        assert len(images) == 2

    def test_case_insensitive_extension(self, tmp_image_dir):
        # Write a valid PNG to a .PNG path
        make_png(tmp_image_dir / "mol.PNG")
        images = db.collect_images(tmp_image_dir, {".png"}, recursive=False)
        assert len(images) == 1


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------

class TestValidateImage:

    def setup_method(self):
        self.log = logging.getLogger("test_validate")

    def test_valid_png_returns_none(self, tmp_path):
        img = make_png(tmp_path / "ok.png")
        assert db.validate_image(img, min_bytes=1024, log=self.log) is None

    def test_valid_jpeg_returns_none(self, tmp_path):
        img = make_jpeg(tmp_path / "ok.jpg")
        assert db.validate_image(img, min_bytes=1024, log=self.log) is None

    def test_file_too_small(self, tmp_path):
        img = make_small_file(tmp_path / "tiny.png")
        result = db.validate_image(img, min_bytes=1024, log=self.log)
        assert result is not None
        assert "too small" in result

    def test_invalid_png_magic(self, tmp_path):
        img = make_bad_magic_png(tmp_path / "bad.png")
        result = db.validate_image(img, min_bytes=1024, log=self.log)
        assert result is not None
        assert "PNG" in result

    def test_invalid_jpeg_magic(self, tmp_path):
        img = make_bad_magic_jpeg(tmp_path / "bad.jpg")
        result = db.validate_image(img, min_bytes=1024, log=self.log)
        assert result is not None
        assert "JPEG" in result

    def test_missing_file(self, tmp_path):
        missing = tmp_path / "ghost.png"
        result = db.validate_image(missing, min_bytes=1024, log=self.log)
        assert result is not None
        assert "cannot stat" in result

    def test_non_png_jpg_extension_skips_magic_check(self, tmp_path):
        # .webp with random bytes — no magic check, only size check
        img = tmp_path / "mol.webp"
        img.write_bytes(b"\x00" * 4096)
        assert db.validate_image(img, min_bytes=1024, log=self.log) is None

    def test_custom_min_size_zero_accepts_small_file(self, tmp_path):
        img = make_small_file(tmp_path / "tiny.png")
        # Override magic check by making it a valid PNG header
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 6)
        result = db.validate_image(img, min_bytes=0, log=self.log)
        assert result is None


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

class TestRunInference:

    def setup_method(self):
        self.log = logging.getLogger("test_inference")

    def test_success_returns_smiles(self, tmp_path):
        img = make_png(tmp_path / "mol.png")
        predict = MagicMock(return_value="CCO")
        smiles, elapsed, error = db.run_inference(predict, img, retries=1, log=self.log)
        assert smiles == "CCO"
        assert error is None
        assert elapsed >= 0

    def test_calls_predict_once_on_success(self, tmp_path):
        img = make_png(tmp_path / "mol.png")
        predict = MagicMock(return_value="CCO")
        db.run_inference(predict, img, retries=3, log=self.log)
        predict.assert_called_once()

    def test_retries_on_failure(self, tmp_path):
        img = make_png(tmp_path / "mol.png")
        predict = MagicMock(side_effect=[RuntimeError("GPU error"), "CCO"])
        smiles, _, error = db.run_inference(predict, img, retries=2, log=self.log)
        assert smiles == "CCO"
        assert error is None
        assert predict.call_count == 2

    def test_returns_error_after_all_retries_exhausted(self, tmp_path):
        img = make_png(tmp_path / "mol.png")
        predict = MagicMock(side_effect=RuntimeError("persistent error"))
        smiles, elapsed, error = db.run_inference(predict, img, retries=3, log=self.log)
        assert smiles == ""
        assert error is not None
        assert "persistent error" in error
        assert predict.call_count == 3

    def test_memory_error_not_retried(self, tmp_path):
        img = make_png(tmp_path / "mol.png")
        predict = MagicMock(side_effect=MemoryError())
        smiles, _, error = db.run_inference(predict, img, retries=3, log=self.log)
        assert smiles == ""
        assert "MemoryError" in error
        predict.assert_called_once()  # No retries

    def test_elapsed_is_non_negative(self, tmp_path):
        img = make_png(tmp_path / "mol.png")
        predict = MagicMock(return_value="c1ccccc1")
        _, elapsed, _ = db.run_inference(predict, img, retries=1, log=self.log)
        assert elapsed >= 0.0

    def test_empty_smiles_on_error(self, tmp_path):
        img = make_png(tmp_path / "mol.png")
        predict = MagicMock(side_effect=ValueError("bad input"))
        smiles, _, error = db.run_inference(predict, img, retries=1, log=self.log)
        assert smiles == ""
        assert error is not None


# ---------------------------------------------------------------------------
# Integration: main() via subprocess-level CLI tests
# ---------------------------------------------------------------------------

def run_main(args: list[str], predict_return=None, predict_side_effect=None):
    """
    Call db.main() with sys.argv patched, DECIMER mocked, and capture exit code.
    Returns (exit_code, mock_predict).
    """
    mock_predict = MagicMock()
    if predict_side_effect is not None:
        mock_predict.side_effect = predict_side_effect
    else:
        mock_predict.return_value = predict_return or "CCO"

    with patch("sys.argv", ["decimer_batch.py"] + args):
        with patch.dict("sys.modules", {"DECIMER": MagicMock(predict_SMILES=mock_predict)}):
            with patch("decimer_batch.db.predict_SMILES", mock_predict, create=True):
                # Patch the lazy import inside main()
                with patch("builtins.__import__", side_effect=_make_importer(mock_predict)):
                    try:
                        db.main()
                        return 0, mock_predict
                    except SystemExit as exc:
                        return exc.code, mock_predict


def _make_importer(mock_predict):
    """Return a custom __import__ that intercepts 'from DECIMER import predict_SMILES'."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _import(name, *args, **kwargs):
        if name == "DECIMER":
            mod = types.ModuleType("DECIMER")
            mod.predict_SMILES = mock_predict
            return mod
        return real_import(name, *args, **kwargs)

    return _import


class TestMainIntegration:
    """Integration tests that invoke main() directly with mocked DECIMER."""

    def _run(self, args, predict_fn=None, tmp_path=None):
        """
        Run main() with patched sys.argv and a mocked predict_SMILES.
        Returns (exit_code, csv_path, mock).
        """
        mock_predict = MagicMock(return_value="CCO") if predict_fn is None else predict_fn

        decimer_mod = types.ModuleType("DECIMER")
        decimer_mod.predict_SMILES = mock_predict

        with patch("sys.argv", ["decimer_batch.py"] + args):
            with patch.dict(sys.modules, {"DECIMER": decimer_mod}):
                try:
                    db.main()
                    code = 0
                except SystemExit as exc:
                    code = exc.code

        return code, mock_predict

    def test_exit_0_on_success(self, tmp_image_dir):
        make_png(tmp_image_dir / "mol.png")
        out = tmp_image_dir / "out.csv"
        code, _ = self._run([str(tmp_image_dir), "-o", str(out)], tmp_path=tmp_image_dir)
        assert code == 0

    def test_exit_1_on_missing_dir(self, tmp_path):
        code, _ = self._run([str(tmp_path / "no_such_dir")])
        assert code == 1

    def test_exit_1_on_no_images(self, tmp_image_dir):
        (tmp_image_dir / "notes.txt").write_text("not an image")
        code, _ = self._run([str(tmp_image_dir)])
        assert code == 1

    def test_exit_2_when_inference_fails(self, tmp_image_dir):
        make_png(tmp_image_dir / "mol.png")
        out = tmp_image_dir / "out.csv"
        bad_predict = MagicMock(side_effect=RuntimeError("model crash"))
        code, _ = self._run([str(tmp_image_dir), "-o", str(out)], predict_fn=bad_predict)
        assert code == 2

    def test_csv_written_on_success(self, tmp_image_dir):
        make_png(tmp_image_dir / "mol.png")
        out = tmp_image_dir / "out.csv"
        self._run([str(tmp_image_dir), "-o", str(out)])
        assert out.exists()
        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 1
        assert rows[0]["smiles"] == "CCO"
        assert rows[0]["status"] == "ok"

    def test_csv_contains_error_row_on_failure(self, tmp_image_dir):
        make_png(tmp_image_dir / "mol.png")
        out = tmp_image_dir / "out.csv"
        bad_predict = MagicMock(side_effect=RuntimeError("oops"))
        self._run([str(tmp_image_dir), "-o", str(out)], predict_fn=bad_predict)
        rows = list(csv.DictReader(out.open()))
        assert rows[0]["smiles"] == ""
        assert rows[0]["status"].startswith("error:")

    def test_csv_contains_skipped_row_for_invalid_image(self, tmp_image_dir):
        make_small_file(tmp_image_dir / "tiny.png")
        out = tmp_image_dir / "out.csv"
        self._run([str(tmp_image_dir), "-o", str(out)])
        rows = list(csv.DictReader(out.open()))
        assert rows[0]["status"].startswith("skipped:")

    def test_csv_fieldnames_correct(self, tmp_image_dir):
        make_png(tmp_image_dir / "mol.png")
        out = tmp_image_dir / "out.csv"
        self._run([str(tmp_image_dir), "-o", str(out)])
        reader = csv.DictReader(out.open())
        assert set(reader.fieldnames) == {"filename", "smiles", "status", "elapsed_s", "timestamp"}

    def test_multiple_images_all_in_csv(self, tmp_image_dir):
        for i in range(3):
            make_png(tmp_image_dir / f"mol_{i}.png")
        out = tmp_image_dir / "out.csv"
        self._run([str(tmp_image_dir), "-o", str(out)])
        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 3

    def test_log_file_created(self, tmp_image_dir, tmp_path):
        make_png(tmp_image_dir / "mol.png")
        log_file = tmp_path / "run.log"
        out = tmp_image_dir / "out.csv"
        self._run([str(tmp_image_dir), "-o", str(out), "--log", str(log_file)])
        assert log_file.exists()
        content = log_file.read_text()
        assert "DECIMER batch processor started" in content

    def test_log_file_contains_smiles(self, tmp_image_dir, tmp_path):
        make_png(tmp_image_dir / "mol.png")
        log_file = tmp_path / "run.log"
        out = tmp_image_dir / "out.csv"
        self._run([str(tmp_image_dir), "-o", str(out), "--log", str(log_file)])
        content = log_file.read_text()
        assert "CCO" in content

    def test_log_file_contains_summary(self, tmp_image_dir, tmp_path):
        make_png(tmp_image_dir / "mol.png")
        log_file = tmp_path / "run.log"
        out = tmp_image_dir / "out.csv"
        self._run([str(tmp_image_dir), "-o", str(out), "--log", str(log_file)])
        content = log_file.read_text()
        assert "Run complete" in content
        assert "Succeeded" in content

    def test_recursive_flag_finds_nested_image(self, tmp_image_dir):
        sub = tmp_image_dir / "sub"
        sub.mkdir()
        make_png(sub / "nested.png")
        out = tmp_image_dir / "out.csv"
        code, _ = self._run([str(tmp_image_dir), "-o", str(out), "-r"])
        assert code == 0
        rows = list(csv.DictReader(out.open()))
        assert any("nested.png" in r["filename"] for r in rows)

    def test_ext_flag_filters_to_specified_type(self, tmp_image_dir):
        make_png(tmp_image_dir / "mol.png")
        make_jpeg(tmp_image_dir / "mol.jpg")
        out = tmp_image_dir / "out.csv"
        self._run([str(tmp_image_dir), "-o", str(out), "--ext", "png"])
        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 1
        assert rows[0]["filename"].endswith(".png")

    def test_retries_flag_passed_to_inference(self, tmp_image_dir):
        make_png(tmp_image_dir / "mol.png")
        out = tmp_image_dir / "out.csv"
        # Fail twice, succeed on 3rd attempt
        flaky = MagicMock(side_effect=[RuntimeError("e1"), RuntimeError("e2"), "CCO"])
        code, _ = self._run(
            [str(tmp_image_dir), "-o", str(out), "--retries", "3"],
            predict_fn=flaky,
        )
        assert code == 0
        assert flaky.call_count == 3

    def test_keyboard_interrupt_exits_130(self, tmp_image_dir):
        make_png(tmp_image_dir / "mol.png")
        out = tmp_image_dir / "out.csv"
        interrupt = MagicMock(side_effect=KeyboardInterrupt())
        code, _ = self._run([str(tmp_image_dir), "-o", str(out)], predict_fn=interrupt)
        assert code == 130

    def test_keyboard_interrupt_partial_csv_saved(self, tmp_image_dir):
        # First image succeeds, second raises KeyboardInterrupt
        for i in range(2):
            make_png(tmp_image_dir / f"mol_{i}.png")
        out = tmp_image_dir / "out.csv"
        responses = ["CCO", KeyboardInterrupt()]
        flaky = MagicMock(side_effect=responses)
        self._run([str(tmp_image_dir), "-o", str(out)], predict_fn=flaky)
        # The first result should still be in the CSV
        rows = list(csv.DictReader(out.open()))
        assert any(r["smiles"] == "CCO" for r in rows)

    def test_default_csv_name_contains_timestamp(self, tmp_image_dir):
        """When --output is omitted, a timestamped CSV is created in image_dir."""
        make_png(tmp_image_dir / "mol.png")
        self._run([str(tmp_image_dir)])
        csvs = list(tmp_image_dir.glob("decimer_results_*.csv"))
        assert len(csvs) == 1

    def test_elapsed_s_is_numeric(self, tmp_image_dir):
        make_png(tmp_image_dir / "mol.png")
        out = tmp_image_dir / "out.csv"
        self._run([str(tmp_image_dir), "-o", str(out)])
        rows = list(csv.DictReader(out.open()))
        assert float(rows[0]["elapsed_s"]) >= 0

    def test_timestamp_column_is_iso_format(self, tmp_image_dir):
        from datetime import datetime
        make_png(tmp_image_dir / "mol.png")
        out = tmp_image_dir / "out.csv"
        self._run([str(tmp_image_dir), "-o", str(out)])
        rows = list(csv.DictReader(out.open()))
        ts = rows[0]["timestamp"]
        # Should parse without error
        datetime.fromisoformat(ts)
