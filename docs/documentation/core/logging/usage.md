---
icon: lucide/book-open-check
title: "Tutorial"
tags:
    - logging
    - usage
    - tutorial
    - core
---

# Logger usage

Whenever a Perseo package or application needs to log, it should use the ``logger`` object exported from ``perseo_core``:

```python title="Basic usage"
from perseo_core import logger

logger.info("Processing started")
logger.warning("Low disk space")
logger.error("Operation failed")
logger.fail("Validation failed")
logger.success("All checks passed")
```

This always operates on the single ``"perseo"`` logger instance. No explicit logger creation or boilerplate is needed in library code.

## Initialization

The logger is **silent by default** (a ``NullHandler`` is attached). A host application must call [`initialize`][perseo_core.logger.initialize] once at startup to enable output:

```python title="Logger initialization"
import logging

from perseo_core import logger

# Minimal — console only
logger.initialize()

# With a log file (parent directories created automatically)
logger.initialize(log_file="logs/app.log", log_level=logging.DEBUG)

# Production — INFO level
logger.initialize(log_file="/var/log/perseo/app.log", log_level=logging.INFO)
```

After initialization, two Rich console handlers (stdout / stderr) are added, plus an optional plain-text file handler. Rich tracebacks are also installed globally.

!!! warning

    Call [`initialize`][perseo_core.logger.initialize] **only once** from your application entry point. Calling it a second time removes all existing handlers and re-creates them.


## Changing log level at runtime

```python title="Changing level"
import logging

from perseo_core import logger

logger.set_level(logging.WARNING)   # suppress DEBUG / INFO
logger.set_level(logging.DEBUG)     # re-enable verbose output
```

### Custom levels

| Level    | Value | Description                                 |
|----------|-------|---------------------------------------------|
| TRACE    | 5     | Hyper-detailed debugging, below DEBUG       |
| DEBUG    | 10    | Standard debug messages *(stdlib)*          |
| INFO     | 20    | Informational messages *(stdlib)*           |
| FAIL     | 21    | Validation or test failure *(custom)*       |
| SUCCESS  | 22    | Validation or test success *(custom)*       |
| WARNING  | 30    | Warning messages *(stdlib)*                 |
| ERROR    | 40    | Error messages *(stdlib)*                   |
| CRITICAL | 50    | Critical errors *(stdlib)*                  |

```python title="Custom levels"
from perseo_core import logger

logger.trace("Entering hot path")
logger.fail("Orbit validation failed")
logger.success("All calibration parameters OK")
```

## Console output

When the output stream is a terminal, Rich renders log messages with colors, styling, and markup support:

```python title="Console output"
from perseo_core import logger

logger.info("[bold green]Pipeline[/] finished [italic]successfully[/]")
```

==Rich markup is parsed only on console output==. File output is always plain text.

!!! note

    If you need literal square brackets in a log message, escape them:

    ```python title=""
    logger.info(r"\[critical\] temperature exceeded \[42\]")  # raw string
    ```


File logging
------------

File output uses the format:
```
| LEVELNAME  @ module_name                 | YYYY-MM-DD HH:MM:SS,mmm | message
```

Example:

```
| INFO      @ my_module                    | 2025-06-01 14:30:00,123 | Processing started
| ERROR     @ my_module                    | 2025-06-01 14:30:01,456 | Operation failed
```

The [`CustomFileHandler`][perseo_core.logger.CustomFileHandler] writes in UTF-8 encoding. Parent directories of the log file path are created automatically.

## Accessing the underlying logger

If you need to add custom handlers, filters, or modify the logger directly:

```python title="Low level logger customization"
from perseo_core.logger import get_logger

perseo_logger = get_logger()
perseo_logger.addHandler(my_custom_handler)
perseo_logger.addFilter(my_custom_filter)
```

This is also the *recommended way to integrate with third-party logging
configurations*.
