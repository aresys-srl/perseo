Logging
=======

The ``perseo_core`` package provides a centralized logging framework that should be used
by all Perseo sub-packages and applications. It is built on top of the standard
:py:mod:`logging` module and the `Rich <https://rich.readthedocs.io/>`_ library for
colorized console output.

Key features:

* Custom log levels: :data:`TRACE`, :data:`FAIL`, :data:`SUCCESS`
* Automatic stream separation:
  ``TRACE`` / ``DEBUG`` / ``INFO`` / ``WARNING`` / ``FAIL`` / ``SUCCESS`` → **stdout**
  ``ERROR`` / ``CRITICAL`` → **stderr**
* TTY-aware console output — Rich colors and markup on terminals, plain text otherwise
* Plain text file logging (never markup in files)
* Rich tracebacks on unhandled exceptions
* Logger object convenience (``logger.info()``, ``logger.fail()``, etc.)
* Silent by default (``NullHandler``) until :func:`initialize_logger` is called

Usage
-----

Whenever a Perseo package or application needs to log, it should use the ``logger``
object exported from ``perseo_core``:

.. code-block:: python

   from perseo_core import logger

   logger.info("Processing started")
   logger.warning("Low disk space")
   logger.error("Operation failed")
   logger.fail("Validation failed")
   logger.success("All checks passed")

This always operates on the single ``"perseo"`` logger instance. No explicit
logger creation or boilerplate is needed in library code.

Initialization
--------------

The logger is **silent by default** (a ``NullHandler`` is attached). A host application
must call :func:`initialize_logger` once at startup to enable output:

.. code-block:: python

   import logging
   from perseo_core import initialize_logger

   # Minimal — console only
   initialize_logger()

   # With a log file (parent directories created automatically)
   initialize_logger(log_file="logs/app.log", log_level=logging.DEBUG)

   # Production — INFO level
   initialize_logger(log_file="/var/log/perseo/app.log", log_level=logging.INFO)

After initialization, two Rich console handlers (stdout / stderr) are added, plus an
optional plain-text file handler. Rich tracebacks are also installed globally.

.. warning::

   Call :func:`initialize_logger` **only once** from your application entry point.
   Calling it a second time removes all existing handlers and re-creates them.

Changing log level at runtime
-----------------------------

.. code-block:: python

   from perseo_core import set_log_level
   import logging

   set_log_level(logging.WARNING)   # suppress DEBUG / INFO
   set_log_level(logging.DEBUG)     # re-enable verbose output

Custom log levels
-----------------

=========== ======= ===========================================
Level       Value   Description
=========== ======= ===========================================
TRACE       5       Hyper-detailed debugging, below DEBUG
DEBUG       10      Standard debug messages *(stdlib)*
INFO        20      Informational messages *(stdlib)*
FAIL        21      Validation or test failure *(custom)*
SUCCESS     22      Validation or test success *(custom)*
WARNING     30      Warning messages *(stdlib)*
ERROR       40      Error messages *(stdlib)*
CRITICAL    50      Critical errors *(stdlib)*
=========== ======= ===========================================

.. code-block:: python

   from perseo_core import logger

   logger.trace("Entering hot path")
   logger.fail("Orbit validation failed")
   logger.success("All calibration parameters OK")

Console output
--------------

When the output stream is a terminal, Rich renders log messages with colors, styling,
and markup support:

.. code-block:: python

   from perseo_core import logger

   logger.info("[bold green]Pipeline[/] finished [italic]successfully[/]")

Rich markup is parsed only on console output. File output is always plain text.

.. note::

   If you need literal square brackets in a log message, escape them:

   .. code-block:: python

       logger.info(r"\[critical\] temperature exceeded \[42\]")  # raw string

File logging
------------

File output uses the format::

    | LEVELNAME  @ module_name                 | YYYY-MM-DD HH:MM:SS,mmm | message

Example::

    | INFO      @ my_module                    | 2025-06-01 14:30:00,123 | Processing started
    | ERROR     @ my_module                    | 2025-06-01 14:30:01,456 | Operation failed

The :class:`CustomFileHandler` writes in UTF-8 encoding. Parent directories of the log
file path are created automatically.

Accessing the underlying logger
-------------------------------

If you need to add custom handlers, filters, or modify the logger directly:

.. code-block:: python

   from perseo_core import get_logger

   logger = get_logger()
   logger.addHandler(my_custom_handler)
   logger.addFilter(my_custom_filter)

This is also the recommended way to integrate with third-party logging
configurations.

Common use cases
----------------

**1. Application entry point**

.. code-block:: python

   # main.py
   import logging
   from perseo_core import logger, initialize_logger

   def main():
       initialize_logger(log_file="perseo.log", log_level=logging.INFO)
       logger.info("Application started")
       # ... your application logic ...

   if __name__ == "__main__":
       main()

**2. Library / sub-package module**

.. code-block:: python

   # perseo_quality/processing.py
   from perseo_core import logger

   def run_validation(data):
       logger.info("Starting validation")
       logger.debug(f"Input data shape: {data.shape}")
       if not data.valid:
           logger.error("Invalid data received")
           raise ValueError("Invalid data")
       logger.info("Validation passed")

**3. Adding context information**

.. code-block:: python

   import logging
   from perseo_core import get_logger

   logger = get_logger()
   adapter = logging.LoggerAdapter(logger, {"request_id": "abc-123"})
   adapter.info("Processing request")  # extra context appended

**4. Suppressing all output**

.. code-block:: python

   # Simply never call initialize_logger().
   # The NullHandler keeps the logger silent.
   from perseo_core import logger

   logger.critical("This goes nowhere")

**5. Console-only output for CLI tools**

.. code-block:: python

   from perseo_core import initialize_logger

   initialize_logger()  # no log_file → console only
