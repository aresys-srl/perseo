# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Utilities for spatially filtering and downsampling raster data.

The filtering and the output transformation are injected, so that different kernels and
transformations can be combined without modifying this module. The data is prepared by
the transformation (e.g. magnitude extraction for amplitude-based transformations),
then spatially filtered by the kernel, and finally transformed by the transformation
into the quick look.
"""

from __future__ import annotations

from perseo_core.imaging.models import SARDataModel, validate_sar_data_model
from perseo_core.imaging.quicklook.kernels import BoxKernel, QuicklookKernel
from perseo_core.imaging.quicklook.transforms import DEFAULT_TRANSFORM, OutputTransform

__all__ = ["quicklook_rendering"]

DEFAULT_KERNEL: QuicklookKernel = BoxKernel()


def quicklook_rendering(
    data: SARDataModel,
    lines_subsampling_factor: float,
    samples_subsampling_factor: float,
    *,
    kernel: QuicklookKernel = DEFAULT_KERNEL,
    transform: OutputTransform = DEFAULT_TRANSFORM,
) -> SARDataModel:
    """Core algorithm for generating the quick look.

    Parameters
    ----------
    data : SARDataModel
        SAR data model to be filtered, with ``azimuth`` and ``range`` dimensions
    lines_subsampling_factor : float
        subsampling factor along azimuth direction
    samples_subsampling_factor : float
        subsampling factor along range direction
    kernel : QuicklookKernel, optional
        filtering kernel, by default BoxKernel()
    transform : OutputTransform, optional
        output transformation applied around the spatial filtering, by default
        percentile clipping

    Returns
    -------
    SARDataModel
        generated quick look, with subsampled axis coordinates

    """
    validate_sar_data_model(data)

    prepared_data = transform.pre_filter(data)
    filtered_data = kernel.apply(prepared_data, lines_subsampling_factor, samples_subsampling_factor)

    return transform.post_filter(filtered_data)
