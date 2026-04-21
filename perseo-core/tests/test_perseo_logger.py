# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unit tests for perseo_logger module."""

import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class LoggerTestCase(unittest.TestCase):
    """Base test case with common logger setup and teardown."""

    def setUp(self):
        """Clear handlers and get logger instance."""
        self.logger = get_logger_object()
        for handler in self.logger.handlers[:]:
            if not isinstance(handler, logging.NullHandler):
                self.logger.removeHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        """Clean up all non-null handlers."""
        for handler in self.logger.handlers[:]:
            if not isinstance(handler, logging.NullHandler):
                self.logger.removeHandler(handler)


class TestCustomLogLevels(unittest.TestCase):
    """Test custom log levels FAIL and SUCCESS."""

    def test_fail_level_value(self):
        """Test that FAIL level is set to 21."""
        self.assertEqual(FAIL, 21)

    def test_success_level_value(self):
        """Test that SUCCESS level is set to 22."""
        self.assertEqual(SUCCESS, 22)

    def test_fail_level_name(self):
        """Test that FAIL level name is registered."""
        self.assertEqual(logging.getLevelName(FAIL), "FAIL")

    def test_success_level_name(self):
        """Test that SUCCESS level name is registered."""
        self.assertEqual(logging.getLevelName(SUCCESS), "SUCCESS")


class TestModuleLevelFunctions(LoggerTestCase):
    """Test module-level logging functions."""

    def setUp(self):
        """Set up test logger with handler."""
        super().setUp()
        # Set up string stream for capturing all output
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger.addHandler(self.handler)

    def test_debug_function(self):
        """Test debug module-level function."""
        debug("Debug test message")
        self.assertIn("Debug test message", self.stream.getvalue())

    def test_info_function(self):
        """Test info module-level function."""
        info("Info test message")
        self.assertIn("Info test message", self.stream.getvalue())

    def test_warning_function(self):
        """Test warning module-level function."""
        warning("Warning test message")
        self.assertIn("Warning test message", self.stream.getvalue())

    def test_error_function(self):
        """Test error module-level function."""
        error("Error test message")
        self.assertIn("Error test message", self.stream.getvalue())

    def test_critical_function(self):
        """Test critical module-level function."""
        critical("Critical test message")
        self.assertIn("Critical test message", self.stream.getvalue())

    def test_fail_function(self):
        """Test fail module-level function."""
        fail("Fail test message")
        self.assertIn("Fail test message", self.stream.getvalue())

    def test_success_function(self):
        """Test success module-level function."""
        success("Success test message")
        self.assertIn("Success test message", self.stream.getvalue())


class TestStreamSeparation(LoggerTestCase):
    """Test that log messages go to correct streams."""

    def setUp(self):
        """Set up test logger with stdout and stderr handlers."""
        super().setUp()

        # Set up streams
        self.stdout_stream = io.StringIO()
        self.stderr_stream = io.StringIO()

        # Add actual perseo handlers
        self.stdout_handler = StdOutConsoleHandler()
        self.stderr_handler = StdErrConsoleHandler()

        self.stdout_handler.stream = self.stdout_stream
        self.stderr_handler.stream = self.stderr_stream

        self.logger.addHandler(self.stdout_handler)
        self.logger.addHandler(self.stderr_handler)

    def test_debug_goes_to_stdout(self):
        """Test that DEBUG messages go to stdout."""
        self.logger.debug("Debug message")
        self.assertIn("Debug message", self.stdout_stream.getvalue())
        self.assertEqual("", self.stderr_stream.getvalue())

    def test_info_goes_to_stdout(self):
        """Test that INFO messages go to stdout."""
        self.logger.info("Info message")
        self.assertIn("Info message", self.stdout_stream.getvalue())
        self.assertEqual("", self.stderr_stream.getvalue())

    def test_warning_goes_to_stdout(self):
        """Test that WARNING messages go to stdout."""
        self.logger.warning("Warning message")
        self.assertIn("Warning message", self.stdout_stream.getvalue())
        self.assertEqual("", self.stderr_stream.getvalue())

    def test_error_goes_to_stderr(self):
        """Test that ERROR messages go to stderr."""
        self.logger.error("Error message")
        self.assertIn("Error message", self.stderr_stream.getvalue())
        self.assertEqual("", self.stdout_stream.getvalue())

    def test_critical_goes_to_stderr(self):
        """Test that CRITICAL messages go to stderr."""
        self.logger.critical("Critical message")
        self.assertIn("Critical message", self.stderr_stream.getvalue())
        self.assertEqual("", self.stdout_stream.getvalue())

    def test_fail_goes_to_stdout(self):
        """Test that FAIL messages go to stdout."""
        self.logger.log(FAIL, "Fail message")
        self.assertIn("Fail message", self.stdout_stream.getvalue())
        self.assertEqual("", self.stderr_stream.getvalue())

    def test_success_goes_to_stdout(self):
        """Test that SUCCESS messages go to stdout."""
        self.logger.log(SUCCESS, "Success message")
        self.assertIn("Success message", self.stdout_stream.getvalue())
        self.assertEqual("", self.stderr_stream.getvalue())


class TestGetLoggerObject(unittest.TestCase):
    """Test get_logger_object function."""

    def test_returns_logger(self):
        """Test that get_logger_object returns a Logger instance."""
        logger = get_logger_object()
        self.assertIsInstance(logger, logging.Logger)

    def test_returns_perseo_logger(self):
        """Test that get_logger_object returns the perseo logger."""
        logger = get_logger_object()
        self.assertEqual(logger.name, "perseo")

    def test_returns_same_instance(self):
        """Test that get_logger_object returns the same instance."""
        logger1 = get_logger_object()
        logger2 = get_logger_object()
        self.assertIs(logger1, logger2)


class TestInitializeLogger(LoggerTestCase):
    """Test initialize_logger function."""

    def test_initialize_without_file(self):
        """Test initialize_logger without file handler."""
        initialize_logger(log_file=None, log_level=logging.INFO)

        # Check that stdout and stderr handlers are added
        handlers = [h for h in self.logger.handlers if not isinstance(h, logging.NullHandler)]
        handler_types = [type(h).__name__ for h in handlers]

        self.assertIn("StdOutConsoleHandler", handler_types)
        self.assertIn("StdErrConsoleHandler", handler_types)

    def test_initialize_with_file(self):
        """Test initialize_logger with file handler."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

        try:
            initialize_logger(log_file=temp_file, log_level=logging.DEBUG)

            # Check handlers
            handlers = [h for h in self.logger.handlers if not isinstance(h, logging.NullHandler)]
            handler_types = [type(h).__name__ for h in handlers]

            self.assertIn("StdOutConsoleHandler", handler_types)
            self.assertIn("StdErrConsoleHandler", handler_types)
            self.assertIn("CustomFileHandler", handler_types)

            # Close handlers to release file
            for handler in handlers:
                if isinstance(handler, CustomFileHandler):
                    handler.close()
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_initialize_sets_log_level(self):
        """Test that initialize_logger sets the correct log level."""
        initialize_logger(log_level=logging.WARNING)
        self.assertEqual(self.logger.level, logging.WARNING)


class TestStdOutConsoleHandler(unittest.TestCase):
    """Test StdOutConsoleHandler."""

    def test_filters_error_and_above(self):
        """Test that StdOutConsoleHandler filters ERROR and CRITICAL."""
        stream = io.StringIO()
        handler = StdOutConsoleHandler()
        handler.stream = stream

        logger = logging.getLogger("test_stdout")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # DEBUG should be logged
        logger.debug("Debug message")
        self.assertIn("Debug message", stream.getvalue())

        # ERROR should not be logged (filtered)
        stream.truncate(0)
        stream.seek(0)
        logger.error("Error message")
        self.assertNotIn("Error message", stream.getvalue())

        logger.removeHandler(handler)

    @patch("sys.stdout.isatty", return_value=True)
    def test_uses_colorful_formatter(self, mock_isatty):
        """Test that handler uses ColorfulFormatter when isatty returns True."""
        handler = StdOutConsoleHandler()
        self.assertIsInstance(handler.formatter, ColorfulFormatter)

    @patch("sys.stdout.isatty", return_value=False)
    def test_uses_plain_formatter(self, mock_isatty):
        """Test that handler uses PlainFormatter when isatty returns False."""
        handler = StdOutConsoleHandler()
        self.assertIsInstance(handler.formatter, PlainFormatter)


class TestStdErrConsoleHandler(unittest.TestCase):
    """Test StdErrConsoleHandler."""

    def test_filters_below_error(self):
        """Test that StdErrConsoleHandler only logs ERROR and above."""
        stream = io.StringIO()
        handler = StdErrConsoleHandler()
        handler.stream = stream

        logger = logging.getLogger("test_stderr")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # INFO should not be logged (filtered)
        logger.info("Info message")
        self.assertNotIn("Info message", stream.getvalue())

        # ERROR should be logged
        logger.error("Error message")
        self.assertIn("Error message", stream.getvalue())

        logger.removeHandler(handler)


class TestCustomFileHandler(unittest.TestCase):
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

            # Check file contents
            with open(temp_file, "r") as f:
                content = f.read()
            self.assertIn("Test message", content)

            logger.removeHandler(handler)
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_uses_plain_formatter(self):
        """Test that handler uses PlainFormatter."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

        try:
            handler = CustomFileHandler(temp_file)
            self.assertIsInstance(handler.formatter, PlainFormatter)
            handler.close()
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestFormatters(unittest.TestCase):
    """Test formatter classes."""

    def test_colorful_formatter_caches_formatters(self):
        """Test that ColorfulFormatter caches formatters."""
        formatter = ColorfulFormatter()
        self.assertIsInstance(formatter._formatters, dict)
        self.assertEqual(len(formatter._formatters), 7)  # 7 log levels

    def test_plain_formatter_caches_formatters(self):
        """Test that PlainFormatter caches formatters."""
        formatter = PlainFormatter()
        self.assertIsInstance(formatter._formatters, dict)
        self.assertEqual(len(formatter._formatters), 7)  # 7 log levels

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
        self.assertIn("Test message", output)

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
        # Check for ANSI escape code presence
        self.assertTrue("\x1b[" in output or "\033[" in output, "ANSI codes should be present in colored output")

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
        # Check that no ANSI escape codes are present
        self.assertNotIn("\x1b[", output, "ANSI escape codes should NOT be in plain output")
        self.assertNotIn("\033[", output, "ANSI escape codes should NOT be in plain output")

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
        self.assertIn("Test message", output)


if __name__ == "__main__":
    unittest.main()
