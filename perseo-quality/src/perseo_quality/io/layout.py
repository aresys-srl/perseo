# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""SAR Product Layout data models"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from itertools import compress

import numpy as np
import numpy.typing as npt
from arepytools.timing.precisedatetime import PreciseDateTime

from perseo_quality.core.custom_errors import (
    AzimuthExceedsBoundariesError,
    InvalidBurstIdError,
    RangeExceedsBoundariesError,
)
from perseo_quality.io.protocol_utilities import roi_validation

# TODO: move to PERSEO Core


@dataclass(frozen=True)
class L1BurstLayout:
    """L1 Burst layout definition and manager"""

    burst_id: int
    lines: int
    samples: int
    lines_step: InitVar[float]
    lines_start: InitVar[PreciseDateTime]
    samples_step: InitVar[float]
    samples_start: InitVar[float]
    azimuth_axis: np.ndarray = field(init=False)
    range_axis: npt.NDArray[np.floating] = field(init=False)
    mid_burst_azimuth: PreciseDateTime = field(init=False)
    mid_burst_range: float = field(init=False)

    def __post_init__(self, lines_step: float, lines_start: PreciseDateTime, samples_step: float, samples_start: float):
        relative_az_axis = np.arange(0, self.lines, 1) * lines_step
        azimuth_axis = relative_az_axis + lines_start
        range_axis = np.arange(0, self.samples, 1) * samples_step + samples_start
        mid_burst_azimuth = azimuth_axis[0] + (azimuth_axis[-1] - azimuth_axis[0]) / 2
        mid_burst_range = range_axis[0] + (range_axis[-1] - range_axis[0]) / 2
        super().__setattr__("azimuth_axis", azimuth_axis)
        super().__setattr__("range_axis", range_axis)
        super().__setattr__("mid_burst_azimuth", mid_burst_azimuth)
        super().__setattr__("mid_burst_range", mid_burst_range)
        super().__setattr__("_relative_az_axis", relative_az_axis)
        super().__setattr__("_lines_step", lines_step)
        super().__setattr__("_samples_step", samples_step)

    def is_azimuth_in_burst(self, az: PreciseDateTime) -> bool:
        """Checking if input azimuth belongs to the burst azimuth axis.

        Parameters
        ----------
        az : PreciseDateTime
            azimuth to be checked

        Returns
        -------
        bool
            True if azimuth belongs to the current burst azimuth axis, else False
        """
        return self.azimuth_axis[0] <= az <= self.azimuth_axis[-1]

    def is_range_in_burst(self, rng: float) -> bool:
        """Checking if input range belongs to the burst range axis.

        Parameters
        ----------
        rng : float
            range to be checked

        Returns
        -------
        bool
            True if range belongs to the current burst range axis, else False
        """
        return self.range_axis[0] <= rng <= self.range_axis[-1]

    def is_azimuth_relative_pixel_in_burst(self, az_burst_pixel: float) -> bool:
        """Checking if input azimuth relative pixel index belongs to the burst azimuth axis.

        Parameters
        ----------
        az_burst_pixel : float
            azimuth pixel index to be checked

        Returns
        -------
        bool
            True if azimuth pixel index belongs to the current burst azimuth axis, else False
        """
        return 0 <= az_burst_pixel < self.lines

    def is_range_relative_pixel_in_burst(self, rng_burst_pixel: float) -> bool:
        """Checking if input range relative pixel index belongs to the burst range axis.

        Parameters
        ----------
        rng_burst_pixel : float
            range pixel index to be checked

        Returns
        -------
        bool
            True if range pixel index belongs to the current burst range axis, else False
        """
        return 0 <= rng_burst_pixel < self.samples

    def azimuth_to_burst_pixel(self, az: PreciseDateTime) -> float:
        """Associating input azimuth with corresponding pixel relative to burst azimuth start.

        Parameters
        ----------
        az : PreciseDateTime
            input azimuth value

        Returns
        -------
        float
            pixel index (sub-pixel precision) corresponding to the input azimuth, relative to the burst azimuth start

        Raises
        ------
        AzimuthExceedsBoundariesError
            if input azimuth exceeds the current burst azimuth axis
        """
        if not self.is_azimuth_in_burst(az):
            raise AzimuthExceedsBoundariesError(f"azimuth {az} exceeds burst boundaries")
        return (az - self.azimuth_axis[0]) / self._lines_step

    def range_to_burst_pixel(self, rng: float) -> float:
        """Associating input range with corresponding pixel relative to burst range start.

        Parameters
        ----------
        rng : float
            input range value

        Returns
        -------
        float
            pixel index (sub-pixel precision) corresponding to the input range, relative to the burst range start

        Raises
        ------
        RangeExceedsBoundariesError
            if input range exceeds the current burst range axis
        """
        if not self.is_range_in_burst(rng):
            raise RangeExceedsBoundariesError(f"range {rng} exceeds burst boundaries")
        return (rng - self.range_axis[0]) / self._samples_step

    def burst_pixel_to_azimuth(self, az_burst_pixel: float) -> PreciseDateTime:
        """Associating input azimuth relative pixel (sub-pixel precision accepted) to corresponding azimuth value.

        Parameters
        ----------
        az_burst_pixel : float
            relative azimuth pixel

        Returns
        -------
        PreciseDateTime
            azimuth associated to the input azimuth relative pixel index

        Raises
        ------
        AzimuthExceedsBoundariesError
            if input azimuth pixel index exceeds the current burst azimuth axis
        """
        if not self.is_azimuth_relative_pixel_in_burst(az_burst_pixel):
            raise AzimuthExceedsBoundariesError(f"azimuth pixel {az_burst_pixel} exceeds burst boundaries")
        return (
            np.interp(az_burst_pixel, np.arange(0, self.azimuth_axis.size), self._relative_az_axis)
            + self.azimuth_axis[0]
        )

    def burst_pixel_to_range(self, rng_burst_pixel: float) -> float:
        """Associating input range relative pixel (sub-pixel precision accepted) to corresponding range value.

        Parameters
        ----------
        rng_burst_pixel : float
            relative range pixel

        Returns
        -------
        float
            range associated to the input range relative pixel index

        Raises
        ------
        RangeExceedsBoundariesError
            if input range pixel index exceeds the current burst range axis
        """
        if not self.is_range_relative_pixel_in_burst(rng_burst_pixel):
            raise RangeExceedsBoundariesError(f"range pixel {rng_burst_pixel} exceeds burst boundaries")
        return np.interp(rng_burst_pixel, np.arange(0, self.range_axis.size), self.range_axis)

    def burst_pixels_to_coordinates(
        self, az_burst_pixel: float, rng_burst_pixel: float
    ) -> tuple[PreciseDateTime, float]:
        """Associating input azimuth and range relative pixels (sub-pixel precision accepted) to corresponding
        coordinates.

        Parameters
        ----------
        az_burst_pixel : float
            relative azimuth pixel
        rng_burst_pixel : float
            relative range pixel

        Returns
        -------
        PreciseDateTime
            azimuth associated to the input azimuth relative pixel index
        float
            range associated to the input range relative pixel index
        """
        return self.burst_pixel_to_azimuth(az_burst_pixel), self.burst_pixel_to_range(rng_burst_pixel)

    def coordinates_to_burst_pixels(self, az: PreciseDateTime, rng: float) -> tuple[float, float]:
        """Associating input azimuth and range values to corresponding pixel indexes.

        Parameters
        ----------
        az : PreciseDateTime
            input azimuth value
        rng : float
            input range value

        Returns
        -------
        float
            relative azimuth pixel associated to the input azimuth value
        float
            relative range pixel associated to the input range value
        """
        return self.azimuth_to_burst_pixel(az), self.range_to_burst_pixel(rng)

    # TODO: repr and print representation to be improved, same for RasterLayout


@dataclass(frozen=True)
class L1RasterLayout:
    """Raster layout definition and management as a collection of BurstLayout objects"""

    lines: int
    samples: int
    bursts: list[L1BurstLayout]
    burst_ids: list[int] = field(init=False)
    burst_starting_line_offsets: np.ndarray = field(init=False)
    raster_azimuth_axis: np.ndarray = field(init=False)
    raster_range_axis: npt.NDArray[np.floating] = field(init=False)
    mid_swath_azimuth: PreciseDateTime = field(init=False)
    mid_swath_range: PreciseDateTime = field(init=False)

    def __post_init__(self):
        """Computing raster axes and mid swath values"""
        if not self.bursts:
            raise ValueError("At least one L1BurstLayout must be specified")
        burst_ids = [b.burst_id for b in self.bursts]
        burst_starting_line_offset = np.array([0] + list(np.cumsum([b.lines for b in self.bursts][:-1])))
        # axes computation
        raster_azimuth_axis = np.concatenate([b.azimuth_axis for b in self.bursts])
        relative_raster_azimuth_axis = np.array(raster_azimuth_axis - raster_azimuth_axis[0]).astype(float)
        mid_swath_azimuth = raster_azimuth_axis[0] + (raster_azimuth_axis[-1] - raster_azimuth_axis[0]) / 2
        # NOTE: only burst with same range axes are considered here
        raster_range_axis = self.bursts[0].range_axis
        mid_swath_range = self.bursts[0].mid_burst_range
        super().__setattr__("burst_ids", burst_ids)
        super().__setattr__("burst_starting_line_offsets", burst_starting_line_offset)
        super().__setattr__("raster_azimuth_axis", raster_azimuth_axis)
        super().__setattr__("raster_range_axis", raster_range_axis)
        super().__setattr__("mid_swath_azimuth", mid_swath_azimuth)
        super().__setattr__("mid_swath_range", mid_swath_range)
        super().__setattr__("_relative_raster_azimuth_axis", relative_raster_azimuth_axis)

    def _get_burst_by_burst_id(self, burst_id: int) -> L1BurstLayout:
        """Selecting the burst layout object corresponding to the input burst id.

        Parameters
        ----------
        burst_id : int
            selected burst id

        Returns
        -------
        L1BurstLayout
            burst layout object corresponding to the selected burst id

        Raises
        ------
        InvalidBurstIdError
            if selected burst id is invalid
        """
        if burst_id not in self.burst_ids:
            raise InvalidBurstIdError(f"Invalid burst id {burst_id}, valid values are {self.burst_ids}")
        return next(filter(lambda x: x.burst_id == burst_id, self.bursts))

    def _get_burst_starting_line_offset_by_burst_id(self, burst_id: int) -> int:
        """Selecting the proper starting line offset based on burst id.

        Parameters
        ----------
        burst_id : int
            selected burst id

        Returns
        -------
        int
            starting line offset for the selected burst id

        Raises
        ------
        InvalidBurstIdError
            if input burst id is not available
        """
        if burst_id not in self.burst_ids:
            raise InvalidBurstIdError(f"selected burst id {burst_id} is not a valid burst")
        burst_list_index = self.burst_ids.index(burst_id)
        return self.burst_starting_line_offsets[burst_list_index]

    def get_burst_layout(self, burst_id: int) -> L1BurstLayout:
        """Returning the burst layout object corresponding to the selected burst id.

        Parameters
        ----------
        burst_id : int
            selected burst id

        Returns
        -------
        L1BurstLayout
            associated burst layout object
        """
        return self._get_burst_by_burst_id(burst_id=burst_id)

    def get_burst_start_coordinates(self, burst_id: int) -> tuple[PreciseDateTime, float]:
        """Returning the azimuth and range start coordinates for the selected burst.

        Parameters
        ----------
        burst_id : int
            burst number

        Returns
        -------
        PreciseDateTime
            azimuth start for the selected burst
        float
            range start for the selected burst
        """
        selected_burst = self._get_burst_by_burst_id(burst_id)
        return selected_burst.azimuth_axis[0], selected_burst.range_axis[0]

    def get_burst_lines(self, burst_id: int) -> int:
        """Returning the number of lines for the selected burst.

        Parameters
        ----------
        burst_id : int
            burst number

        Returns
        -------
        int
            azimuth lines for the selected burst
        """
        selected_burst = self._get_burst_by_burst_id(burst_id)
        return selected_burst.lines

    def get_burst_samples(self, burst_id: int) -> int:
        """Returning the number of samples for the selected burst.

        Parameters
        ----------
        burst_id : int
            burst number

        Returns
        -------
        int
            range samples for the selected burst
        """
        selected_burst = self._get_burst_by_burst_id(burst_id)
        return selected_burst.samples

    def is_azimuth_in_raster(self, az: PreciseDateTime) -> bool:
        """Check if input azimuth does not exceeds the raster azimuth boundaries.

        Parameters
        ----------
        az : PreciseDateTime
            azimuth to be checked

        Returns
        -------
        bool
            True if azimuth belongs to the raster azimuth axis, else False
        """
        return self.raster_azimuth_axis[0] <= az <= self.raster_azimuth_axis[-1]

    def is_range_in_raster(self, rng: float) -> bool:
        """Check if input range does not exceeds the raster range boundaries.

        Parameters
        ----------
        rng : float
            range to be checked

        Returns
        -------
        bool
            True if range belongs to the raster range axis, else False
        """
        return self.raster_range_axis[0] <= rng <= self.raster_range_axis[-1]

    def azimuth_to_bursts_association(self, az: PreciseDateTime) -> list[int]:
        """Detecting which bursts intersect the input azimuth.

        Parameters
        ----------
        az : PreciseDateTime
            input azimuth

        Returns
        -------
        list[int]
            list of burst ids intersecting the input azimuth

        Raises
        ------
        AzimuthExceedsBoundariesError
            if azimuth exceeds raster boundaries
        """
        if not self.is_azimuth_in_raster(az=az):
            raise AzimuthExceedsBoundariesError(f"azimuth {az} exceeds raster boundaries")
        burst_check_mask = [b.is_azimuth_in_burst(az) for b in self.bursts]
        return list(compress(self.burst_ids, burst_check_mask))

    def range_to_bursts_association(self, rng: float) -> list[int]:
        """Detecting which bursts intersect the input range.

        Parameters
        ----------
        rng : float
            input range

        Returns
        -------
        list[int]
            list of burst ids intersecting the input range

        Raises
        ------
        RangeExceedsBoundariesError
            if range exceeds raster boundaries
        """
        if not self.is_range_in_raster(rng):
            raise RangeExceedsBoundariesError(f"range {rng} exceeds raster boundaries")
        burst_check_mask = [b.is_range_in_burst(rng) for b in self.bursts]
        return list(compress(self.burst_ids, burst_check_mask))

    def azimuth_to_pixel_conversion(self, az: PreciseDateTime, burst_id: int | None = None) -> list[tuple[int, float]]:
        """Converting input azimuth to raster pixel with sub-pixel precision.

        Parameters
        ----------
        az : PreciseDateTime
            input azimuth
        burst_id : int | None, optional
            burst to be selected in case of overlapping bursts for the input azimuth, if None all the pixels
            associated to the given azimuth are returned, by default None

        Returns
        -------
        list[tuple[int, float]]
            list of (burst_id, pixel) corresponding to the input azimuth, more than one tuple in case of overlapping
            bursts and no burst id provided as input

        Raises
        ------
        InvalidBurstIdError
            if the selected burst_id does not contain the input azimuth
        """
        bursts_of_interest = self.azimuth_to_bursts_association(az)
        if burst_id is not None:
            if burst_id not in bursts_of_interest:
                raise InvalidBurstIdError(f"selected burst id {burst_id} does not contain input azimuth")
            bursts_of_interest = [burst_id]
        pixel_indexes = []
        for burst in bursts_of_interest:
            burst_layout = self._get_burst_by_burst_id(burst_id=burst)
            pixel_indexes.append(
                (
                    burst,
                    burst_layout.azimuth_to_burst_pixel(az)
                    + self._get_burst_starting_line_offset_by_burst_id(burst_id=burst),
                )
            )
        return pixel_indexes

    def range_to_pixel_conversion(self, rng: float, burst_id: int | None = None) -> list[tuple[int, float]]:
        """Converting input range to raster pixel with sub-pixel precision.

        Parameters
        ----------
        rng : float
            input range
        burst_id : int | None, optional
            burst to be selected in case of overlapping bursts for the input range, if None all the pixels
            associated to the given range are returned, by default None

        Returns
        -------
        list[tuple[int, float]]
            list of (burst_id, pixel) corresponding to the input range, more than one tuple in case of
            more burst covering the input range and no burst id selected

        Raises
        ------
        InvalidBurstIdError
            if the selected burst_id does not contain the input range
        """
        bursts_of_interest = self.range_to_bursts_association(rng)
        if burst_id is not None:
            if burst_id not in bursts_of_interest:
                raise InvalidBurstIdError(f"input burst id {burst_id} does not contain input range")
            bursts_of_interest = [burst_id]
        pixel_indexes = []
        for burst in bursts_of_interest:
            # NOTE: the offset is set to 0, supposed same range axis for all burst
            pixel_indexes.append(
                (
                    burst,
                    self._get_burst_by_burst_id(burst_id=burst).range_to_burst_pixel(rng) + 0,
                )
            )
        return pixel_indexes

    def pixel_to_azimuth_conversion(self, az_pixel_index: float) -> PreciseDateTime:
        """Converting input azimuth pixel index to raster azimuth.

        Parameters
        ----------
        az_pixel_index : float
            input azimuth pixel index, sub-pixel precision is supported

        Returns
        -------
        PreciseDateTime
            raster azimuth corresponding to the input azimuth pixel index

        Raises
        ------
        AzimuthExceedsBoundariesError
            if input azimuth pixel index exceeds azimuth raster boundaries
        """
        if az_pixel_index < 0 or az_pixel_index >= self.lines:
            raise AzimuthExceedsBoundariesError(f"azimuth index {az_pixel_index} exceeds raster boundaries")
        return (
            np.interp(az_pixel_index, np.arange(0, self.lines), self._relative_raster_azimuth_axis)
            + self.raster_azimuth_axis[0]
        )

    def pixel_to_range_conversion(self, rng_pixel_index: float) -> float:
        """Converting input range pixel index to raster range.

        Parameters
        ----------
        rng_pixel_index : int
            input range pixel index, sub-pixel precision is supported

        Returns
        -------
        float
            raster range corresponding to the input range pixel index

        Raises
        ------
        RangeExceedsBoundariesError
            if input range pixel index exceeds range raster boundaries
        """
        if rng_pixel_index < 0 or rng_pixel_index >= self.samples:
            raise RangeExceedsBoundariesError(f"range index {rng_pixel_index} exceeds raster boundaries")
        return np.interp(rng_pixel_index, np.arange(0, self.samples), self.raster_range_axis)

    def is_roi_in_raster(self, roi: list[int, int, int, int], burst: int | None = None) -> bool:
        """Checking if input roi is fully within the raster boundaries.

        Parameters
        ----------
        roi : list[int, int, int, int]
            [reading start line, reading start sample, number of lines to be read, number of samples to be read]
        burst : int | None, optional
            if provided the roi is checked against the burst boundaries and not the raster ones, by default None

        Returns
        -------
        bool
            True if roi lies within raster or burst boundaries, else False
        """
        burst_boundaries = None
        if burst:
            burst_boundaries = [
                self.burst_starting_line_offsets[burst],
                0,
                self.bursts[burst].lines,
                self.bursts[burst].samples,
            ]
        try:
            roi_validation(
                roi=roi, raster_boundaries=[0, 0, self.lines, self.samples], burst_boundaries=burst_boundaries
            )
            return True
        except (AzimuthExceedsBoundariesError, RangeExceedsBoundariesError):
            return False
