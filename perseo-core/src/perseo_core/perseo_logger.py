# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Centralized logging framework for Perseo with custom levels and colorized output.

This module provides a pre-configured logger for the Perseo framework with support for
custom log levels (FAIL and SUCCESS), colorized console output, and structured file logging.

The logger separates output streams:
  - DEBUG, INFO, WARNING -> stdout
  - ERROR, CRITICAL, FAIL -> stderr

Module-level functions provide convenient access to the logger without explicit initialization.
Custom log levels can be enabled programmatically, and handlers can be customized as needed.

The logger has a null handler by default so that by default perseo framework is silent.

Custom Log Levels
-----------------
These levels are designed to indicate the outcome of a validation test.
FAIL : 21
    Indicates a failed test.
SUCCESS : 22
    Indicates a successful test.

Examples
--------
Basic usage with module-level functions:

    from perseo_core import perseo_logger

    perseo_logger.initialize_logger(log_file="perseo.log", log_level=logging.INFO)

    perseo_logger.info("Processing started")
    perseo_logger.error("An error occurred")
    perseo_logger.fail("Operation failed")
    perseo_logger.success("Operation completed")

Bypass initialization and add a different file handler :

    from perseo_core.perseo_logger import get_logger_object

    get_logger_object().addHandler(handler)

Notes
-----
The logger is initialized with both stdout and stderr handlers by default.
To customize handlers or log levels, access the underlying logger via `_PERSEO_LOGGER`
that you can access via `get_logger_object()`.
"""

from __future__ import annotations

import logging
import sys

FAIL = 21
SUCCESS = 22
logging.addLevelName(FAIL, "FAIL")
logging.addLevelName(SUCCESS, "SUCCESS")

AnsiColors = {
    "GREY": "\x1b[38;20m",
    "YELLOW": "\x1b[33;20m",
    "HIGHLIGHT_YELLOW": "\033[48;5;226m",
    "GREEN": "\x1b[1;32m",
    "HIGHLIGHT_GREEN": "\033[48;5;40m",
    "RED": "\x1b[31;20m",
    "HIGHLIGHT_RED": "\033[48;5;196m",
    "BOLD_RED": "\x1b[31;1m",
    "PURPLE": "\x1b[1;35m",
    "BLUE": "\x1b[1;34m",
    "LIGHT_BLUE": "\x1b[1;36m",
    "BOLD": "\x1b[1m",
    "UNDERLINE": "\x1b[4m",
    "RESET": "\x1b[0m",
}


fmt = "| %(levelname)-9s @ %(module)s| %(asctime)s | %(message)s"

_colors = {
    logging.DEBUG: AnsiColors["GREY"],
    logging.INFO: AnsiColors["GREY"],
    logging.WARNING: AnsiColors["YELLOW"],
    logging.ERROR: AnsiColors["RED"],
    logging.CRITICAL: AnsiColors["BOLD_RED"],
    FAIL: AnsiColors["BOLD"] + AnsiColors["HIGHLIGHT_RED"],
    SUCCESS: AnsiColors["BOLD"] + AnsiColors["HIGHLIGHT_GREEN"],
}


def get_format_table(fmt: str, *, colors: dict[int, str] | None) -> dict[int, str]:
    """Generate format strings for each log level, with optional colors."""
    levels = (
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
        FAIL,
        SUCCESS,
    )
    if colors is None:
        return {level: fmt for level in levels}
    return {level: colors[level] + fmt + AnsiColors["RESET"] for level in levels}


FILE_FORMATS = get_format_table(fmt, colors=None)
CONSOLE_FORMATS = get_format_table(fmt, colors=_colors)


class ColorfulFormatter(logging.Formatter):
    """Formatter with ANSI colors for console output."""

    def __init__(self):
        super().__init__()
        self._formatters = {level: logging.Formatter(fmt) for level, fmt in CONSOLE_FORMATS.items()}

    def format(self, record):
        formatter = self._formatters.get(record.levelno)
        return formatter.format(record) if formatter else str(record.getMessage())


class PlainFormatter(logging.Formatter):
    """Plain text formatter without colors for file output."""

    def __init__(self):
        super().__init__()
        self._formatters = {level: logging.Formatter(fmt) for level, fmt in FILE_FORMATS.items()}

    def format(self, record):
        formatter = self._formatters.get(record.levelno)
        return formatter.format(record) if formatter else str(record.getMessage())


class StdOutConsoleHandler(logging.StreamHandler):
    """Custom logging stream handler for stdout (non-error messages)."""

    def __init__(self):
        super().__init__(stream=sys.stdout)
        formatter = ColorfulFormatter() if sys.stdout.isatty() else PlainFormatter()
        self.setFormatter(formatter)
        self.addFilter(lambda record: record.levelno < logging.ERROR)


class StdErrConsoleHandler(logging.StreamHandler):
    """Custom logging stream handler for stderr (error messages)."""

    def __init__(self):
        super().__init__(stream=sys.stderr)
        formatter = ColorfulFormatter() if sys.stderr.isatty() else PlainFormatter()
        self.setFormatter(formatter)
        self.addFilter(lambda record: record.levelno >= logging.ERROR)


class CustomFileHandler(logging.FileHandler):
    """Custom logging file handler to centralize logging."""

    def __init__(self, filename):
        """File handler to write log to disk.

        Parameters
        ----------
        filename : str or Path-like
            log filename
        """
        super().__init__(filename=filename)
        self.setFormatter(PlainFormatter())


_PERSEO_LOGGER = logging.getLogger("perseo")
_PERSEO_LOGGER.addHandler(logging.NullHandler())

debug = _PERSEO_LOGGER.debug
info = _PERSEO_LOGGER.info
warning = _PERSEO_LOGGER.warning
error = _PERSEO_LOGGER.error
critical = _PERSEO_LOGGER.critical


def fail(msg, *args, **kwargs):
    return _PERSEO_LOGGER.log(FAIL, msg, *args, **kwargs)


def success(msg, *args, **kwargs):
    return _PERSEO_LOGGER.log(SUCCESS, msg, *args, **kwargs)


def initialize_logger(log_file: str | None = None, log_level: int = logging.DEBUG):
    """Initialize the Perseo logger with console and optional file handlers.

    Parameters
    ----------
    log_file : str or Path-like, optional
        If provided, a file handler will be added to log to this file.
    log_level : int, optional
        Logging level to set for the logger (default is logging.DEBUG).
    """
    for handler in _PERSEO_LOGGER.handlers[:]:
        handler.close()
    _PERSEO_LOGGER.handlers.clear()

    _PERSEO_LOGGER.propagate = False

    _PERSEO_LOGGER.setLevel(log_level)
    _PERSEO_LOGGER.addHandler(StdOutConsoleHandler())
    _PERSEO_LOGGER.addHandler(StdErrConsoleHandler())
    if log_file is not None:
        _PERSEO_LOGGER.addHandler(CustomFileHandler(log_file))


def get_logger_object() -> logging.Logger:
    """Get the Perseo logger instance."""
    return _PERSEO_LOGGER
