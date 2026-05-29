# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
This module provides a pre-configured logger for the Perseo framework with support for
custom log levels (TRACE, FAIL, SUCCESS), Rich-formatted console output, and plain file logging.

The logger is designed to be flexible and easy to use, with sensible defaults for console output
and optional file logging. By default, it uses Rich handlers to provide visually appealing and
informative log messages in the console, while file output is plain text for compatibility.

This module is the primary way to log throughout the Perseo ecosystem.

To initialize the logger with sensible defaults, use the ``initialize`` function.
Otherwise, the logger has a null handler by default and it will be silent until configured.

For advanced usage, you can access the underlying logger instance via the ``get_logger()`` function
and customize handlers, formatters, or log levels as needed.

### Custom Log Levels

- TRACE (5): hyper-detailed debugging, below DEBUG.
- FAIL (21): indicates a failed validation or test.
- SUCCESS (22): indicates a successful validation or test.

### Console log default output streams

The logger separates output streams:

   - TRACE, DEBUG, INFO, WARNING, FAIL, SUCCESS -> stdout
   - ERROR, CRITICAL -> stderr

### Examples

Basic usage:

```python
import logging
from perseo_core import logger

logger.initialize(log_file="perseo.log", log_level=logging.INFO)

logger.info("Processing started")
logger.error("An error occurred")
logger.fail("Operation failed")
logger.success("Operation completed")
```

Bypass initialization and add a different file handler:

```python
from perseo_core.logger import get_logger

get_logger().addHandler(handler)
```

!!! note
    The logger is initialized with both stdout and stderr Rich handlers by default.
    To customize handlers or log levels, access the underlying logger via ``get_logger()``.

"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

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


_PERSEO_LOGGER: logging.Logger = logging.getLogger("perseo")
_PERSEO_LOGGER.addHandler(logging.NullHandler())


def initialize(log_file: str | None = None, log_level: int = logging.DEBUG):
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


def set_level(level: int):
    """Set the logging level for the Perseo logger.

    Parameters
    ----------
    level : int
        One of ``TRACE`` (5), ``logging.DEBUG`` (10), ``logging.INFO`` (20),
        ``FAIL`` (21), ``SUCCESS`` (22), ``logging.WARNING`` (30),
        ``logging.ERROR`` (40), ``logging.CRITICAL`` (50).
    """
    _PERSEO_LOGGER.setLevel(level)


def get_logger() -> logging.Logger:
    """Get the Perseo logger instance."""
    return _PERSEO_LOGGER


info = _PERSEO_LOGGER.info
debug = _PERSEO_LOGGER.debug
warning = _PERSEO_LOGGER.warning
error = _PERSEO_LOGGER.error
critical = _PERSEO_LOGGER.critical


def fail(msg: str, *args: object, **kwargs: Any) -> None:
    return _PERSEO_LOGGER.log(FAIL, msg, *args, **kwargs)


def success(msg: str, *args: object, **kwargs: Any) -> None:
    return _PERSEO_LOGGER.log(SUCCESS, msg, *args, **kwargs)


def trace(msg: str, *args: object, **kwargs: Any) -> None:
    return _PERSEO_LOGGER.log(TRACE, msg, *args, **kwargs)
