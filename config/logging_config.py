"""Logging configuration module."""

import logging
import sys
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def setup_verbose() -> None:
    """Setup verbose logging (INFO level with more detail)."""
    setup_logging(logging.INFO)


def setup_debug() -> None:
    """Setup debug logging (DEBUG level)."""
    setup_logging(logging.DEBUG)
