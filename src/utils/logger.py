"""
Centralized logging configuration for the OCR pipeline project.
"""

import logging
import os
import sys
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record):
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                )
        return super().format(record)


def setup_logging(
    name: str | None = None,
    level: str | None = None,
    log_file: str | None = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Parameters
    ----------
    name : str, optional
        Logger name
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Path to log file
    console : bool
        Log to console (default: True)

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logger.level)
        console_formatter = ColoredFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logger.level)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with standard configuration."""
    log_file = os.getenv("LOG_FILE")
    return setup_logging(name=name, log_file=log_file)
