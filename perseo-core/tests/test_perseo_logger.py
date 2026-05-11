# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unit tests for perseo_logger module."""

import io
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from perseo_core.perseo_logger import (
    FAIL,
    SUCCESS,
    ColorfulFormatter,
    CustomFileHandler,
    PlainFormatter,
    StdErrConsoleHandler,
    StdOutConsoleHandler,
    critical,
    debug,
    error,
    fail,
    get_logger_object,
    info,
    initialize_logger,
    success,
    warning,
)


def _setup_logger():
    """Clear handlers and return logger instance."""
    logger = get_logger_object()
    for handler in logger.handlers[:]:
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


class LoggerTestCase:
    """Base test case with common logger setup and teardown."""

    @pytest.fixture(autouse=True)
    def setup_logger(self):
        self.logger = _setup_logger()
        yield
        for handler in self.logger.handlers[:]:
            if not isinstance(handler, logging.NullHandler):
                self.logger.removeHandler(handler)


class TestCustomLogLevels:
    """Test custom log levels FAIL and SUCCESS."""

    def test_fail_level_value(self):
        """Test that FAIL level is set to 21."""
        assert FAIL == 21

    def test_success_level_value(self):
        """Test that SUCCESS level is set to 22."""
        assert SUCCESS == 22

    def test_fail_level_name(self):
        """Test that FAIL level name is registered."""
        assert logging.getLevelName(FAIL) == "FAIL"

    def test_success_level_name(self):
        """Test that SUCCESS level name is registered."""
        assert logging.getLevelName(SUCCESS) == "SUCCESS"


class TestModuleLevelFunctions(LoggerTestCase):
    """Test module-level logging functions."""

    @pytest.fixture(autouse=True)
    def setup_stream(self, setup_logger):
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger.addHandler(self.handler)

    def test_debug_function(self):
        """Test debug module-level function."""
        debug("Debug test message")
        assert "Debug test message" in self.stream.getvalue()

    def test_info_function(self):
        """Test info module-level function."""
        info("Info test message")
        assert "Info test message" in self.stream.getvalue()

    def test_warning_function(self):
        """Test warning module-level function."""
        warning("Warning test message")
        assert "Warning test message" in self.stream.getvalue()

    def test_error_function(self):
        """Test error module-level function."""
        error("Error test message")
        assert "Error test message" in self.stream.getvalue()

    def test_critical_function(self):
        """Test critical module-level function."""
        critical("Critical test message")
        assert "Critical test message" in self.stream.getvalue()

    def test_fail_function(self):
        """Test fail module-level function."""
        fail("Fail test message")
        assert "Fail test message" in self.stream.getvalue()

    def test_success_function(self):
        """Test success module-level function."""
        success("Success test message")
        assert "Success test message" in self.stream.getvalue()


class TestStreamSeparation(LoggerTestCase):
    """Test that log messages go to correct streams."""

    @pytest.fixture(autouse=True)
    def setup_stream_separation(self, setup_logger):
        self.stdout_stream = io.StringIO()
        self.stderr_stream = io.StringIO()

        self.stdout_handler = StdOutConsoleHandler()
        self.stderr_handler = StdErrConsoleHandler()

        self.stdout_handler.stream = self.stdout_stream
        self.stderr_handler.stream = self.stderr_stream

        self.logger.addHandler(self.stdout_handler)
        self.logger.addHandler(self.stderr_handler)

    def test_debug_goes_to_stdout(self):
        """Test that DEBUG messages go to stdout."""
        self.logger.debug("Debug message")
        assert "Debug message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""

    def test_info_goes_to_stdout(self):
        """Test that INFO messages go to stdout."""
        self.logger.info("Info message")
        assert "Info message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""

    def test_warning_goes_to_stdout(self):
        """Test that WARNING messages go to stdout."""
        self.logger.warning("Warning message")
        assert "Warning message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""

    def test_error_goes_to_stderr(self):
        """Test that ERROR messages go to stderr."""
        self.logger.error("Error message")
        assert "Error message" in self.stderr_stream.getvalue()
        assert self.stdout_stream.getvalue() == ""

    def test_critical_goes_to_stderr(self):
        """Test that CRITICAL messages go to stderr."""
        self.logger.critical("Critical message")
        assert "Critical message" in self.stderr_stream.getvalue()
        assert self.stdout_stream.getvalue() == ""

    def test_fail_goes_to_stdout(self):
        """Test that FAIL messages go to stdout."""
        self.logger.log(FAIL, "Fail message")
        assert "Fail message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""

    def test_success_goes_to_stdout(self):
        """Test that SUCCESS messages go to stdout."""
        self.logger.log(SUCCESS, "Success message")
        assert "Success message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""


class TestGetLoggerObject:
    """Test get_logger_object function."""

    def test_returns_logger(self):
        """Test that get_logger_object returns a Logger instance."""
        logger = get_logger_object()
        assert isinstance(logger, logging.Logger)

    def test_returns_perseo_logger(self):
        """Test that get_logger_object returns the perseo logger."""
        logger = get_logger_object()
        assert logger.name == "perseo"

    def test_returns_same_instance(self):
        """Test that get_logger_object returns the same instance."""
        logger1 = get_logger_object()
        logger2 = get_logger_object()
        assert logger1 is logger2


class TestInitializeLogger(LoggerTestCase):
    """Test initialize_logger function."""

    def test_initialize_without_file(self):
        """Test initialize_logger without file handler."""
        initialize_logger(log_file=None, log_level=logging.INFO)

        handlers = [h for h in self.logger.handlers if not isinstance(h, logging.NullHandler)]
        handler_types = [type(h).__name__ for h in handlers]

        assert "StdOutConsoleHandler" in handler_types
        assert "StdErrConsoleHandler" in handler_types

    def test_initialize_with_file(self):
        """Test initialize_logger with file handler."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

        try:
            initialize_logger(log_file=temp_file, log_level=logging.DEBUG)

            handlers = [h for h in self.logger.handlers if not isinstance(h, logging.NullHandler)]
            handler_types = [type(h).__name__ for h in handlers]

            assert "StdOutConsoleHandler" in handler_types
            assert "StdErrConsoleHandler" in handler_types
            assert "CustomFileHandler" in handler_types

            for handler in handlers:
                if isinstance(handler, CustomFileHandler):
                    handler.close()
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_initialize_sets_log_level(self):
        """Test that initialize_logger sets the correct log level."""
        initialize_logger(log_level=logging.WARNING)
        assert self.logger.level == logging.WARNING


class TestStdOutConsoleHandler:
    """Test StdOutConsoleHandler."""

    def test_filters_error_and_above(self):
        """Test that StdOutConsoleHandler filters ERROR and CRITICAL."""
        stream = io.StringIO()
        handler = StdOutConsoleHandler()
        handler.stream = stream

        logger = logging.getLogger("test_stdout")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.debug("Debug message")
        assert "Debug message" in stream.getvalue()

        stream.truncate(0)
        stream.seek(0)
        logger.error("Error message")
        assert "Error message" not in stream.getvalue()

        logger.removeHandler(handler)

    @patch("sys.stdout.isatty", return_value=True)
    def test_uses_colorful_formatter(self, mock_isatty):
        """Test that handler uses ColorfulFormatter when isatty returns True."""
        handler = StdOutConsoleHandler()
        assert isinstance(handler.formatter, ColorfulFormatter)

    @patch("sys.stdout.isatty", return_value=False)
    def test_uses_plain_formatter(self, mock_isatty):
        """Test that handler uses PlainFormatter when isatty returns False."""
        handler = StdOutConsoleHandler()
        assert isinstance(handler.formatter, PlainFormatter)


class TestStdErrConsoleHandler:
    """Test StdErrConsoleHandler."""

    def test_filters_below_error(self):
        """Test that StdErrConsoleHandler only logs ERROR and above."""
        stream = io.StringIO()
        handler = StdErrConsoleHandler()
        handler.stream = stream

        logger = logging.getLogger("test_stderr")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("Info message")
        assert "Info message" not in stream.getvalue()

        logger.error("Error message")
        assert "Error message" in stream.getvalue()

        logger.removeHandler(handler)


class TestCustomFileHandler:
    """Test CustomFileHandler."""

    def test_creates_file(self):
        """Test that CustomFileHandler creates a log file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

        try:
            handler = CustomFileHandler(temp_file)
            logger = logging.getLogger("test_file")
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

            logger.info("Test message")
            handler.close()

            with open(temp_file, "r") as f:
                content = f.read()
            assert "Test message" in content

            logger.removeHandler(handler)
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_uses_plain_formatter(self):
        """Test that handler uses PlainFormatter."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

        try:
            handler = CustomFileHandler(temp_file)
            assert isinstance(handler.formatter, PlainFormatter)
            handler.close()
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestFormatters:
    """Test formatter classes."""

    def test_colorful_formatter_caches_formatters(self):
        """Test that ColorfulFormatter caches formatters."""
        formatter = ColorfulFormatter()
        assert isinstance(formatter._formatters, dict)
        assert len(formatter._formatters) == 7

    def test_plain_formatter_caches_formatters(self):
        """Test that PlainFormatter caches formatters."""
        formatter = PlainFormatter()
        assert isinstance(formatter._formatters, dict)
        assert len(formatter._formatters) == 7

    def test_colorful_formatter_output(self):
        """Test ColorfulFormatter produces colored output."""
        formatter = ColorfulFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "Test message" in output

    def test_colorful_formatter_includes_ansi_codes(self):
        """Test that ColorfulFormatter includes ANSI escape codes."""
        formatter = ColorfulFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "\x1b[" in output or "\033[" in output, "ANSI codes should be present in colored output"

    def test_plain_formatter_excludes_ansi_codes(self):
        """Test that PlainFormatter does NOT include ANSI escape codes."""
        formatter = PlainFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "\x1b[" not in output, "ANSI escape codes should NOT be in plain output"
        assert "\033[" not in output, "ANSI escape codes should NOT be in plain output"

    def test_plain_formatter_output(self):
        """Test PlainFormatter produces plain output."""
        formatter = PlainFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "Test message" in output
