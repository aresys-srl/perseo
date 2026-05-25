# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
This module provides a pre-configured logger for the Perseo framework with support for
custom log levels (TRACE, FAIL, SUCCESS), Rich-formatted console output, and plain file logging.

The logger separates output streams:

   - TRACE, DEBUG, INFO, WARNING, FAIL, SUCCESS -> stdout
   - ERROR, CRITICAL -> stderr

The ``logger`` object is the primary way to log throughout the Perseo ecosystem.
Custom log levels can be enabled programmatically, and handlers can be customized as needed.

The logger has a null handler by default so that by default the perseo framework is silent.

### Custom Log Levels

- TRACE (5): hyper-detailed debugging, below DEBUG.
- FAIL (21): indicates a failed validation or test.
- SUCCESS (22): indicates a successful validation or test.

### Examples

Basic usage:

```python
from perseo_core import logger
from perseo_core.logger import initialize_logger

initialize_logger(log_file="perseo.log", log_level=20)  # INFO

logger.info("Processing started")
logger.error("An error occurred")
logger.fail("Operation failed")
logger.success("Operation completed")
```

Bypass initialization and add a different file handler:

```python
from perseo_core import get_logger

get_logger().addHandler(handler)
```

!!! note
    The logger is initialized with both stdout and stderr Rich handlers by default.
    To customize handlers or log levels, access the underlying logger via
    ``get_logger()`` or the ``logger`` alias.

"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install as install_rich_tracebacks

TRACE = 5
FAIL = 21
SUCCESS = 22

logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(FAIL, "FAIL")
logging.addLevelName(SUCCESS, "SUCCESS")

_RICH_THEME = Theme(
    {
        "logging.level.trace": "dim white",
        "logging.level.debug": "dim",
        "logging.level.info": "bold",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "logging.level.fail": "bold reverse bright_red",
        "logging.level.success": "bold reverse bright_green",
    }
)

FILE_FORMAT = "| %(levelname)-9s @ %(module)-30.30s| %(asctime)s | %(message)s"


class PerseoLogger(logging.Logger):
    """Logger subclass with trace(), fail(), success() convenience methods."""

    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, kwargs)

    def fail(self, msg, *args, **kwargs):
        if self.isEnabledFor(FAIL):
            self._log(FAIL, msg, args, kwargs)

    def success(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, msg, args, kwargs)


logging.setLoggerClass(PerseoLogger)
_PERSEO_LOGGER = logging.getLogger("perseo")
_PERSEO_LOGGER.addHandler(logging.NullHandler())


class StdOutRichHandler(RichHandler):
    """Rich console handler for stdout (non-error messages)."""

    def __init__(self):
        super().__init__(
            console=Console(file=sys.stdout, theme=_RICH_THEME),
            show_time=True,
            show_path=True,
            show_level=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        self.addFilter(lambda record: record.levelno < logging.ERROR)


class StdErrRichHandler(RichHandler):
    """Rich console handler for stderr (error messages)."""

    def __init__(self):
        super().__init__(
            console=Console(file=sys.stderr, theme=_RICH_THEME),
            show_time=True,
            show_path=True,
            show_level=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        self.addFilter(lambda record: record.levelno >= logging.ERROR)


class PlainFileFormatter(logging.Formatter):
    """Plain text formatter for file output (no Rich markup)."""

    def __init__(self):
        super().__init__(fmt=FILE_FORMAT)


class CustomFileHandler(logging.FileHandler):
    """File handler with plain text formatting and UTF-8 encoding."""

    def __init__(self, filename: str):
        super().__init__(filename=filename, encoding="utf-8")
        self.setFormatter(PlainFileFormatter())


def initialize_logger(log_file: str | None = None, log_level: int = logging.DEBUG):
    """Initialize the Perseo logger with Rich console and optional file handler.

    Parameters
    ----------
    log_file : str or Path-like, optional
        If provided, a file handler will be added to write plain-text logs to this path.
        Parent directories are created automatically if they do not exist.
    log_level : int, optional
        Logging level to set for the logger (default is ``logging.DEBUG``).
    """
    for handler in _PERSEO_LOGGER.handlers[:]:
        _PERSEO_LOGGER.removeHandler(handler)
        handler.close()

    _PERSEO_LOGGER.propagate = False
    _PERSEO_LOGGER.setLevel(log_level)
    _PERSEO_LOGGER.addHandler(StdOutRichHandler())
    _PERSEO_LOGGER.addHandler(StdErrRichHandler())

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _PERSEO_LOGGER.addHandler(CustomFileHandler(str(log_path)))

    install_rich_tracebacks(show_locals=True)


def set_log_level(level: int):
    """Set the logging level for the Perseo logger.

    Parameters
    ----------
    level : int
        One of ``TRACE`` (5), ``logging.DEBUG`` (10), ``logging.INFO`` (20),
        ``logging.WARNING`` (30), ``logging.ERROR`` (40), ``logging.CRITICAL`` (50).
    """
    _PERSEO_LOGGER.setLevel(level)


def get_logger() -> logging.Logger:
    """Get the Perseo logger instance."""
    return _PERSEO_LOGGER


logger: logging.Logger = _PERSEO_LOGGER
