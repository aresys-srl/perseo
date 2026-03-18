---
icon: lucide/stamp
title: "Data Model"
tags:
    - quality
    - data model
    - input
---

# Quality Data Model

All the analyses implemented in this package have been designed to abstract away the dependency on the input format, relying
solely on a generic Python protocol. Once this protocol is satisfied, it enables the execution of all implemented analyses
regardless of the type of input product.

This data model protocol and its utilities are available in the ``perseo_quality.io`` module.

## Protocol

::: perseo_quality.io.quality_input_protocol

::: perseo_quality.io.protocol_utilities

## Layout

::: perseo_quality.io.layout
