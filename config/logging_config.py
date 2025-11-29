"""
Logging configuration module.

Provides centralized logging setup with consistent formatting
and log level management.
"""

import logging
import sys
from types import TracebackType
from typing import List, Optional, Type


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Setup logging configuration for the entire application.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        format_string: Custom format string for log messages
        log_file: Optional path to log file
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # FIX 1: Explicitly type the list as generic logging.Handler objects
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level, format=format_string, handlers=handlers, force=True
    )

    # Reduce verbosity of some third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)
    logging.getLogger("shapely").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter:
    """
    Context manager for temporarily changing log level.

    Example:
        with LoggerAdapter(logger, logging.DEBUG):
            logger.debug("This will be shown")
    """

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level

    def __enter__(self) -> logging.Logger:
        self.logger.setLevel(self.new_level)
        return self.logger

    # FIX 2: Add correct type annotations for the exception arguments
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.logger.setLevel(self.old_level)


# Quick setup for verbose/debug modes
def setup_verbose() -> None:
    """Setup logging for verbose mode (INFO level)."""
    setup_logging(level=logging.INFO)


def setup_debug() -> None:
    """Setup logging for debug mode (DEBUG level)."""
    setup_logging(level=logging.DEBUG)


def setup_quiet() -> None:
    """Setup logging for quiet mode (WARNING level)."""
    setup_logging(level=logging.WARNING)
