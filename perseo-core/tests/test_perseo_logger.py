# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unit tests for perseo_logger module."""

import io
import logging
import tempfile
from pathlib import Path

import pytest
from rich.console import Console
from rich.logging import RichHandler

from perseo_core import logger
from perseo_core.logger import (
    FAIL,
    SUCCESS,
    TRACE,
    CustomFileHandler,
    PlainFileFormatter,
    StdErrRichHandler,
    StdOutRichHandler,
    get_logger,
    initialize,
    set_level,
)


class TestCustomLogLevels:
    """Test custom log levels TRACE, FAIL and SUCCESS."""

    def test_trace_level_value(self):
        assert TRACE == 5

    def test_fail_level_value(self):
        assert FAIL == 21

    def test_success_level_value(self):
        assert SUCCESS == 22

    def test_trace_level_name(self):
        assert logging.getLevelName(TRACE) == "TRACE"

    def test_fail_level_name(self):
        assert logging.getLevelName(FAIL) == "FAIL"

    def test_success_level_name(self):
        assert logging.getLevelName(SUCCESS) == "SUCCESS"


class TestLoggerObjectMethods:
    """Test logging via the logger object."""

    @pytest.fixture(autouse=True)
    def _setup_stream(self):
        self.stream = io.StringIO()
        handler = logging.StreamHandler(self.stream)
        get_logger().addHandler(handler)

    def test_trace(self):
        logger.set_level(TRACE)
        logger.trace("Trace test message")
        assert "Trace test message" in self.stream.getvalue()

    def test_debug(self):
        logger.set_level(logging.DEBUG)
        logger.debug("Debug test message")
        assert "Debug test message" in self.stream.getvalue()

    def test_info(self):
        logger.set_level(logging.INFO)
        logger.info("Info test message")
        assert "Info test message" in self.stream.getvalue()

    def test_warning(self):
        logger.set_level(logging.WARNING)
        logger.warning("Warning test message")
        assert "Warning test message" in self.stream.getvalue()

    def test_error(self):
        logger.set_level(logging.ERROR)
        logger.error("Error test message")
        assert "Error test message" in self.stream.getvalue()

    def test_critical(self):
        logger.set_level(logging.CRITICAL)
        logger.critical("Critical test message")
        assert "Critical test message" in self.stream.getvalue()

    def test_fail(self):
        logger.set_level(FAIL)
        logger.fail("Fail test message")
        assert "Fail test message" in self.stream.getvalue()

    def test_success(self):
        logger.set_level(SUCCESS)
        logger.success("Success test message")
        assert "Success test message" in self.stream.getvalue()


class TestStreamSeparation:
    """Test that log messages go to correct streams."""

    @pytest.fixture(autouse=True)
    def _setup_streams(self):
        self.stdout_stream = io.StringIO()
        self.stderr_stream = io.StringIO()

        stdout_handler = StdOutRichHandler()
        stdout_handler.console = Console(file=self.stdout_stream)

        stderr_handler = StdErrRichHandler()
        stderr_handler.console = Console(file=self.stderr_stream)

        logger = get_logger()
        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)

    def test_trace_goes_to_stdout(self):
        logger.set_level(TRACE)
        logger.trace("Trace message")
        assert "Trace message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""

    def test_debug_goes_to_stdout(self):
        logger.set_level(logging.DEBUG)
        logger.debug("Debug message")
        assert "Debug message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""

    def test_info_goes_to_stdout(self):
        logger.set_level(logging.INFO)
        logger.info("Info message")
        assert "Info message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""

    def test_warning_goes_to_stdout(self):
        logger.set_level(logging.WARNING)
        logger.warning("Warning message")
        assert "Warning message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""

    def test_error_goes_to_stderr(self):
        logger.set_level(logging.ERROR)
        logger.error("Error message")
        assert "Error message" in self.stderr_stream.getvalue()
        assert self.stdout_stream.getvalue() == ""

    def test_critical_goes_to_stderr(self):
        logger.set_level(logging.CRITICAL)
        logger.critical("Critical message")
        assert "Critical message" in self.stderr_stream.getvalue()
        assert self.stdout_stream.getvalue() == ""

    def test_fail_goes_to_stdout(self):
        logger.set_level(FAIL)
        logger.fail("Fail message")
        assert "Fail message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""

    def test_success_goes_to_stdout(self):
        logger.set_level(SUCCESS)
        logger.success("Success message")
        assert "Success message" in self.stdout_stream.getvalue()
        assert self.stderr_stream.getvalue() == ""


class TestGetLogger:
    """Test get_logger function."""

    def test_returns_logger(self):
        logger = get_logger()
        assert isinstance(logger, logging.Logger)

    def test_returns_perseo_logger(self):
        logger = get_logger()
        assert logger.name == "perseo"

    def test_returns_same_instance(self):
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2


class TestInitializeLogger:
    """Test initialize_logger function."""

    def test_initialize_without_file(self):
        initialize(log_file=None, log_level=logging.INFO)

        handlers = [h for h in get_logger().handlers if not isinstance(h, logging.NullHandler)]
        handler_types = [type(h).__name__ for h in handlers]

        assert "StdOutRichHandler" in handler_types
        assert "StdErrRichHandler" in handler_types

    def test_initialize_with_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

        try:
            initialize(log_file=temp_file, log_level=logging.DEBUG)

            handlers = [h for h in get_logger().handlers if not isinstance(h, logging.NullHandler)]
            handler_types = [type(h).__name__ for h in handlers]

            assert "StdOutRichHandler" in handler_types
            assert "StdErrRichHandler" in handler_types
            assert "CustomFileHandler" in handler_types

            for handler in handlers:
                if isinstance(handler, CustomFileHandler):
                    handler.close()
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_initialize_sets_log_level(self):
        initialize(log_level=logging.WARNING)
        assert get_logger().level == logging.WARNING


class TestSetLogLevel:
    """Test set_log_level function."""

    def test_set_log_level(self):
        set_level(logging.ERROR)
        assert get_logger().level == logging.ERROR


class TestStdOutRichHandler:
    """Test StdOutRichHandler."""

    def test_is_rich_handler(self):
        handler = StdOutRichHandler()
        assert isinstance(handler, RichHandler)

    def test_filters_error_and_above(self):
        stream = io.StringIO()
        handler = StdOutRichHandler()
        handler.console = Console(file=stream)

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


class TestStdErrRichHandler:
    """Test StdErrRichHandler."""

    def test_is_rich_handler(self):
        handler = StdErrRichHandler()
        assert isinstance(handler, RichHandler)

    def test_filters_below_error(self):
        stream = io.StringIO()
        handler = StdErrRichHandler()
        handler.console = Console(file=stream)

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
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

        try:
            handler = CustomFileHandler(temp_file)
            assert isinstance(handler.formatter, PlainFileFormatter)
            handler.close()
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestPlainFileFormatter:
    """Test PlainFileFormatter."""

    def test_output_includes_message(self):
        formatter = PlainFileFormatter()
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

    def test_excludes_ansi_codes(self):
        formatter = PlainFileFormatter()
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
