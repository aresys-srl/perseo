# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Protocol utilities and support functionalities for Quality I/O"""

from __future__ import annotations

from perseo_quality.core.custom_errors import AzimuthExceedsBoundariesError, RangeExceedsBoundariesError


def roi_validation(
    roi: list[int, int, int, int],
    raster_boundaries: list[int, int, int, int],
    burst_boundaries: list[int, int, int, int] | None = None,
) -> None:
    """Validating the region of interest to be read from raster.

    Parameters
    ----------
    roi : list[int, int, int, int]
        [reading start line, reading start sample, number of lines to be read, number of samples to be read]
    raster_boundaries : list[int, int, int, int]
        [0, 0, raster number of lines, raster number of samples]
    burst_boundaries : list[int, int, int, int] | None, optional
        [current burst starting line, current burst starting sample, lines of current burst, samples of current burst],
        if None no validation against burst boundaries is performed, by default None

    Raises
    ------
    AzimuthExceedsBoundariesError
        if roi boundaries exceed raster or burst azimuth boundaries
    RangeExceedsBoundariesError
        ir roi boundaries exceed raster or burst range boundaries
    """

    # checking for raster boundaries
    if roi[0] >= raster_boundaries[2] or roi[0] < 0:
        # starting azimuth line to be read is out of azimuth swath boundaries
        raise AzimuthExceedsBoundariesError(f"First ROI line {roi[0]} exceeds azimuth swath boundaries")
    if roi[0] + roi[2] > raster_boundaries[2]:
        # last azimuth line to be read is out of azimuth swath boundaries
        raise AzimuthExceedsBoundariesError(f"Last ROI line {roi[0] + roi[2]} exceeds azimuth swath boundaries")
    if roi[1] >= raster_boundaries[3] or roi[1] < 0:
        # starting range sample to be read is out of range swath boundaries
        raise RangeExceedsBoundariesError(f"First ROI sample {roi[1]} exceeds range swath boundaries")
    if roi[1] + roi[3] > raster_boundaries[3]:
        # last range sample to be read is out of range swath boundaries
        raise RangeExceedsBoundariesError(f"Last ROI sample {roi[1] + roi[3]} exceeds range swath boundaries")

    # checking for burst boundaries
    if burst_boundaries is not None:
        if roi[0] < burst_boundaries[0] or roi[0] >= burst_boundaries[0] + burst_boundaries[2]:
            # starting azimuth line to be read is out of azimuth burst boundaries
            raise AzimuthExceedsBoundariesError(f"First ROI line {roi[0]} exceeds azimuth burst boundaries")
        if roi[0] + roi[2] > burst_boundaries[0] + burst_boundaries[2]:
            # last azimuth line to be read is out of azimuth burst boundaries
            raise AzimuthExceedsBoundariesError(f"Last ROI line {roi[0] + roi[2]} exceeds azimuth burst boundaries")
