# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Logger - Custom Handlers and Formatters
---------------------------------------
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path


class AnsiColors(Enum):
    """Ansi escape color strings for Logging Formatter"""

    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    HIGHLIGHT_YELLOW = "\033[48;5;226m"
    GREEN = "\x1b[1;32m"
    HIGHLIGHT_GREEN = "\033[48;5;40m"
    RED = "\x1b[31;20m"
    HIGHLIGHT_RED = "\033[48;5;196m"
    BOLD_RED = "\x1b[31;1m"
    PURPLE = "\x1b[1;35m"
    BLUE = "\x1b[1;34m"
    LIGHT_BLUE = "\x1b[1;36m"
    BOLD = "\x1b[1m"
    UNDERLINE = "\x1b[4m"
    RESET = "\x1b[0m"


class ConsoleFormatter(logging.Formatter):
    """Custom logger formatter with colors"""

    # message formatting layout
    fmt = "| %(levelname)-9s @ %(module)s| %(asctime)s | %(message)s"

    FORMATS = {
        logging.DEBUG: AnsiColors.GREY.value + fmt + AnsiColors.RESET.value,
        logging.INFO: AnsiColors.GREY.value + fmt + AnsiColors.RESET.value,
        logging.WARNING: AnsiColors.YELLOW.value + fmt + AnsiColors.RESET.value,
        logging.ERROR: AnsiColors.RED.value + fmt + AnsiColors.RESET.value,
        logging.CRITICAL: AnsiColors.BOLD_RED.value + fmt + AnsiColors.RESET.value,
        logging.FAIL: AnsiColors.BOLD.value + AnsiColors.HIGHLIGHT_RED.value + fmt + AnsiColors.RESET.value,
        logging.SUCCESS: AnsiColors.BOLD.value + AnsiColors.HIGHLIGHT_GREEN.value + fmt + AnsiColors.RESET.value,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        return formatter.format(record)


class FileFormatter(logging.Formatter):
    """Custom logger formatter with colors"""

    # message formatting layout
    fmt = "| %(levelname)-9s @ %(module)s| %(asctime)s | %(message)s"

    FORMATS = {
        logging.DEBUG: fmt,
        logging.INFO: fmt,
        logging.WARNING: fmt,
        logging.ERROR: fmt,
        logging.CRITICAL: fmt,
        logging.FAIL: fmt,
        logging.SUCCESS: fmt,
    }

    def __init__(self):
        super().__init__()
        self._fmt = self.fmt

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        return formatter.format(record)


class ConsoleHandler(logging.StreamHandler):
    """Custom logging stream handler to centralize logging"""

    def __init__(self):
        super().__init__(stream=sys.stdout)
        self.setFormatter(ConsoleFormatter())


class CustomFileHandler(logging.FileHandler):
    """Custom logging file handler to centralize logging"""

    def __init__(self, filename: Path):
        """File handler to write log to disk.

        Parameters
        ----------
        filename : Path
            log filename
        """
        super().__init__(filename=filename)
        self.setFormatter(FileFormatter())
