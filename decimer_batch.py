#!/usr/bin/env python
"""
decimer_batch.py
----------------
Run DECIMER inference on all images in a directory and write results to a CSV.
Includes structured logging (console + rotating log file) and robust error handling.

Usage:
    python decimer_batch.py <image_dir> [options]

Examples:
    python decimer_batch.py ~/structures_to_process
    python decimer_batch.py ~/structures_to_process -o ~/results/smiles.csv
    python decimer_batch.py ~/structures_to_process -r --log ~/logs/decimer.log
    python decimer_batch.py ~/structures_to_process --retries 2 --min-size 1024
"""

import argparse
import csv
import logging
import logging.handlers
import sys
import time
from datetime import datetime
from pathlib import Path


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

# Minimum valid image file size in bytes (guards against 0-byte / truncated files)
DEFAULT_MIN_BYTES = 1024


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path | None, verbose: bool) -> logging.Logger:
    """Configure root logger with a console handler and optional rotating file handler."""
    logger = logging.getLogger("decimer_batch")
    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter by level
    logger.handlers.clear()
    logger.propagate = False

    fmt_detailed = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_console = logging.Formatter("%(levelname)-8s %(message)s")

    # Console handler — INFO by default, DEBUG if --verbose
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt_console)
    logger.addHandler(console)

    # Rotating file handler — always DEBUG level, 5 MB per file, 3 backups
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_detailed)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch DECIMER inference: images → SMILES CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing chemical structure images",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: <image_dir>/decimer_results_<timestamp>.csv)",
    )
    parser.add_argument(
        "--log", "-l",
        type=Path,
        default=None,
        metavar="LOG_FILE",
        help="Write detailed logs to this file (rotating, 5 MB × 3 backups)",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=None,
        metavar="EXT",
        help=f"Image extensions to include (default: all supported)",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search image_dir recursively",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        metavar="N",
        help="Retry failed images up to N times before marking as error",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_BYTES,
        metavar="BYTES",
        help="Skip images smaller than this many bytes (likely corrupt/empty)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show DEBUG-level messages on the console",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image collection and validation
# ---------------------------------------------------------------------------

def collect_images(image_dir: Path, extensions: set, recursive: bool) -> list[Path]:
    glob = image_dir.rglob if recursive else image_dir.glob
    return sorted(
        p for p in glob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def validate_image(path: Path, min_bytes: int, log: logging.Logger) -> str | None:
    """
    Basic pre-flight checks on an image file.
    Returns None if the file looks OK, or an error string describing the problem.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"cannot stat file: {exc}"

    if size < min_bytes:
        return f"file too small ({size} bytes < {min_bytes} minimum)"

    # Peek at magic bytes to detect obviously wrong file types
    try:
        with path.open("rb") as fh:
            header = fh.read(12)
    except OSError as exc:
        return f"cannot read file: {exc}"

    # PNG: 8-byte signature
    if path.suffix.lower() == ".png" and not header.startswith(b"\x89PNG"):
        return "invalid PNG header (file may be corrupt or mislabelled)"

    # JPEG: starts with FF D8
    if path.suffix.lower() in (".jpg", ".jpeg") and not header.startswith(b"\xff\xd8"):
        return "invalid JPEG header (file may be corrupt or mislabelled)"

    log.debug("  validation OK: %s (%d bytes)", path.name, size)
    return None


# ---------------------------------------------------------------------------
# Inference with retry
# ---------------------------------------------------------------------------

def run_inference(predict_fn, img_path: Path, retries: int, log: logging.Logger):
    """
    Call predict_fn(img_path) up to `retries` times.
    Returns (smiles, elapsed_s, error_message|None).
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        if attempt > 1:
            log.warning("  Retry %d/%d for %s", attempt, retries, img_path.name)
        t0 = time.time()
        try:
            smiles = predict_fn(str(img_path))
            elapsed = time.time() - t0
            if attempt > 1:
                log.info("  Succeeded on attempt %d", attempt)
            return smiles, elapsed, None
        except MemoryError:
            # Non-recoverable — don't retry
            elapsed = time.time() - t0
            log.error("  MemoryError — skipping retries")
            return "", elapsed, "MemoryError"
        except Exception as exc:
            elapsed = time.time() - t0
            last_exc = exc
            log.debug("  Attempt %d failed after %.1fs: %s", attempt, elapsed, exc, exc_info=True)

    return "", elapsed, str(last_exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve paths
    image_dir = args.image_dir.expanduser().resolve()
    log_path = args.log.expanduser().resolve() if args.log else None

    log = setup_logging(log_path, args.verbose)

    log.info("=" * 60)
    log.info("DECIMER batch processor started")
    log.info("Image directory : %s", image_dir)
    if log_path:
        log.info("Log file        : %s", log_path)

    # Validate input directory
    if not image_dir.is_dir():
        log.error("'%s' is not a directory.", image_dir)
        sys.exit(1)

    # Determine extensions
    extensions = (
        {f".{e.lstrip('.')}" for e in args.ext}
        if args.ext else SUPPORTED_EXTENSIONS
    )
    log.debug("Extensions      : %s", extensions)

    # Collect images
    images = collect_images(image_dir, extensions, args.recursive)
    if not images:
        log.error(
            "No images found in '%s' with extensions %s.",
            image_dir, extensions,
        )
        sys.exit(1)
    log.info("Images found    : %d", len(images))

    # Determine output path
    output_path = args.output or (
        image_dir / f"decimer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Output CSV      : %s", output_path)
    log.info("Retries         : %d", args.retries)
    log.info("Min file size   : %d bytes", args.min_size)
    log.info("-" * 60)

    # Load model
    log.info("Loading DECIMER model (first load may take ~30–60 s)...")
    try:
        from DECIMER import predict_SMILES
    except ImportError as exc:
        log.critical(
            "Failed to import DECIMER: %s\n"
            "Make sure you are running inside the DECIMER conda environment.",
            exc,
        )
        sys.exit(1)
    except Exception as exc:
        log.critical("Unexpected error loading DECIMER: %s", exc, exc_info=True)
        sys.exit(1)
    log.info("Model ready.")
    log.info("-" * 60)

    # Process images
    fieldnames = ["filename", "smiles", "status", "elapsed_s", "timestamp"]
    n_ok = n_skipped = n_failed = 0
    run_start = time.time()

    try:
        with output_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, img_path in enumerate(images, start=1):
                rel = img_path.relative_to(image_dir)
                log.info("[%d/%d] %s", i, len(images), rel)

                # Pre-flight validation
                validation_error = validate_image(img_path, args.min_size, log)
                if validation_error:
                    log.warning("  SKIPPED — %s", validation_error)
                    writer.writerow({
                        "filename": str(rel),
                        "smiles": "",
                        "status": f"skipped: {validation_error}",
                        "elapsed_s": "0.00",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    })
                    csvfile.flush()
                    n_skipped += 1
                    continue

                # Inference
                smiles, elapsed, error = run_inference(
                    predict_SMILES, img_path, args.retries, log
                )

                if error:
                    log.warning("  FAILED (%.1fs) — %s", elapsed, error)
                    status = f"error: {error}"
                    n_failed += 1
                else:
                    log.info("  OK (%.1fs) → %s", elapsed, smiles)
                    status = "ok"
                    n_ok += 1

                writer.writerow({
                    "filename": str(rel),
                    "smiles": smiles,
                    "status": status,
                    "elapsed_s": f"{elapsed:.2f}",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })
                csvfile.flush()

    except KeyboardInterrupt:
        log.warning("Interrupted by user — partial results saved to %s", output_path)
        sys.exit(130)
    except OSError as exc:
        log.critical("Cannot write CSV: %s", exc)
        sys.exit(1)

    # Summary
    total_elapsed = time.time() - run_start
    log.info("=" * 60)
    log.info("Run complete in %.1f s", total_elapsed)
    log.info("  Succeeded : %d", n_ok)
    log.info("  Skipped   : %d", n_skipped)
    log.info("  Failed    : %d", n_failed)
    log.info("  Total     : %d", len(images))
    log.info("Results written to: %s", output_path)
    if log_path:
        log.info("Full log written to: %s", log_path)
    log.info("=" * 60)

    # Non-zero exit if any images failed (useful for scripting)
    if n_failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
