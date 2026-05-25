---
icon: lucide/logs
tags:
    - logging
    - core
---

# Logging { #log data-toc-label="Logging" }

The ``perseo_core`` package provides a centralized logging framework that should be used
by all Perseo sub-packages and applications. It is built on top of the standard
[`logging`](https://docs.python.org/3/library/logging.html) module and the [Rich](https://rich.readthedocs.io/) library for colorized console output.

Key features:

- Custom log levels: `TRACE`, `FAIL`, `SUCCESS`
- Automatic stream separation:  
  ``TRACE`` / ``DEBUG`` / ``INFO`` / ``WARNING`` / ``FAIL`` / ``SUCCESS`` → **stdout**  
  ``ERROR`` / ``CRITICAL`` → **stderr**
- TTY-aware console output — Rich colors and markup on terminals, plain text otherwise
- Plain text file logging (never markup in files)
- Rich tracebacks on unhandled exceptions
- Logger object convenience (``logger.info()``, ``logger.fail()``, etc.)
- Silent by default (``NullHandler``) until [`initialize_logger`][perseo_core.logger.initialize_logger] is called
