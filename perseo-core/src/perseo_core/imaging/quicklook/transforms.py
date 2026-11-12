# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Output transformations for quick-look imagery.

Transforms honor the OutputTransform contract with two phases: pre_filter is applied
to the data before spatial filtering, post_filter to the filtered data. Amplitude-based
transformations (db, clip_percentile) extract the magnitude in pre_filter, so that the
spatial filtering operates on amplitudes; the phase transformation keeps the data as-is,
so that the spatial filtering operates on the complex data and the phase is extracted
afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

    from perseo_core.imaging.models import SARDataModel

__all__ = [
    "TRANSFORMS",
    "ClipPercentileTransform",
    "OutputTransform",
    "PhaseTransform",
    "ToDbTransform",
    "get_transform",
]


class OutputTransform(Protocol):
    """Contract for quick-look output transformations."""

    def pre_filter(self, data: SARDataModel) -> SARDataModel:
        """Prepare the data before spatial filtering.

        Amplitude-based transformations extract the data magnitude, phase-based
        transformations keep the data unchanged.

        Parameters
        ----------
        data : SARDataModel
            data model to prepare

        Returns
        -------
        SARDataModel
            prepared data model

        """
        ...

    def post_filter(self, data: SARDataModel) -> SARDataModel:
        """Transform the filtered data into the final quick look.

        Parameters
        ----------
        data : SARDataModel
            filtered data model

        Returns
        -------
        SARDataModel
            transformed quick look

        """
        ...


def _extract_magnitude(data: SARDataModel) -> SARDataModel:
    """Extract the data magnitude, preserving dimensions, coordinates and attrs."""
    return data.copy(data=np.abs(data.values))


def to_db(data: npt.ArrayLike) -> np.ndarray:
    """Transform amplitude magnitude to dB."""
    magnitude = np.abs(np.asarray(data))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20.0 * np.log10(magnitude)


def to_phase(data: np.ndarray) -> np.ndarray:
    """Transform quick look data to phase."""
    return np.angle(data)


def abs_percentiles_threshold_filtering(
    data: np.ndarray, min_percentile: float = 1.0, max_percentile: float = 99.0
) -> np.ndarray:
    """Clip the input data between min and max percentiles.

    Overwriting the values below the minimum percentile and above the maximum percentile with the values corresponding
    to those percentiles respectively.

    Parameters
    ----------
    data : np.ndarray
        input data to be filtered
    min_percentile : float, optional
        minimum percentile threshold, by default 1
    max_percentile : float, optional
        maximum percentile threshold, by default 99

    Returns
    -------
    np.ndarray
        filtered data

    """
    if not 0 <= min_percentile <= max_percentile <= 100:
        msg = (
            "Percentiles must satisfy 0 <= min_percentile <= max_percentile <= 100."
            f"Got {min_percentile}, {max_percentile}."
        )
        raise ValueError(msg)

    magnitudes = np.abs(np.asarray(data))

    lower, upper = np.nanpercentile(
        magnitudes,
        [min_percentile, max_percentile],
    )

    return np.clip(magnitudes, lower, upper)


def clip_percentile(
    data: np.ndarray,
    *,
    min_percentile: float,
    max_percentile: float,
) -> np.ndarray:
    """Transform quicklook by clipping percentiles.

    Parameters
    ----------
    data : np.ndarray
        quick look data to be clipped
    min_percentile : float
        minimum percentile threshold
    max_percentile : float
        max percentile threshold

    Returns
    -------
    np.ndarray
        clipped quick look data

    """
    return abs_percentiles_threshold_filtering(
        data=data,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
    )


class ToDbTransform:
    """Transform quick look to dB: filters amplitudes, then converts to dB."""

    def pre_filter(self, data: SARDataModel) -> SARDataModel:
        """Extract the data magnitude before spatial filtering."""
        return _extract_magnitude(data)

    def post_filter(self, data: SARDataModel) -> SARDataModel:
        """Convert the filtered amplitudes to dB."""
        return data.copy(data=to_db(data.values))


class PhaseTransform:
    """Transform quick look to phase: filters complex data, then extracts the phase."""

    def pre_filter(self, data: SARDataModel) -> SARDataModel:
        """Keep the data unchanged before spatial filtering."""
        return data

    def post_filter(self, data: SARDataModel) -> SARDataModel:
        """Extract the phase of the filtered complex data."""
        return data.copy(data=to_phase(data.values))


class ClipPercentileTransform:
    """Transform quick look by clipping percentiles: filters amplitudes, then clips."""

    def __init__(self, *, min_percentile: float = 1.0, max_percentile: float = 99.0) -> None:
        """Initialize the percentile clipping transformation.

        Parameters
        ----------
        min_percentile : float, optional
            minimum percentile threshold, by default 1
        max_percentile : float, optional
            maximum percentile threshold, by default 99

        Raises
        ------
        ValueError
            if the percentiles are not ordered

        """
        if not 0 <= min_percentile <= max_percentile <= 100:
            msg = (
                "Percentiles must satisfy 0 <= min_percentile <= max_percentile <= 100."
                f"Got {min_percentile}, {max_percentile}."
            )
            raise ValueError(msg)
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile

    def pre_filter(self, data: SARDataModel) -> SARDataModel:
        """Extract the data magnitude before spatial filtering."""
        return _extract_magnitude(data)

    def post_filter(self, data: SARDataModel) -> SARDataModel:
        """Clip the filtered amplitudes between the percentiles."""
        return data.copy(
            data=abs_percentiles_threshold_filtering(
                data=data.values,
                min_percentile=self._min_percentile,
                max_percentile=self._max_percentile,
            )
        )


DEFAULT_TRANSFORM: OutputTransform = ClipPercentileTransform()

TRANSFORMS: dict[str, Callable[..., OutputTransform]] = {
    "db": ToDbTransform,
    "phase": PhaseTransform,
    "clip_percentile": ClipPercentileTransform,
}


def get_transform(name: str, **params: object) -> OutputTransform:
    """Get an output transformation by name, optionally binding its parameters.

    Parameters
    ----------
    name : str
        name of the transformation, one of the keys of TRANSFORMS
    **params
        parameters passed to the transformation constructor, e.g. min_percentile
        and max_percentile for clip_percentile

    Returns
    -------
    OutputTransform
        transformation with the requested parameters bound

    """
    try:
        transform_class = TRANSFORMS[name]
    except KeyError:
        msg = f"Unknown transform {name}. Available transforms are {sorted(TRANSFORMS)}."
        raise ValueError(msg) from None
    return transform_class(**params)
