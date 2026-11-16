# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Image rendering and saving of quick-look data to disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

__all__ = ["image_scale_clip", "save_image_from_array"]

_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def image_scale_clip(array: npt.NDArray[np.floating], max_scale: float = 255.0) -> npt.NDArray[np.floating]:
    """Clip and scale the input array to [0, 255].

    Parameters
    ----------
    array : npt.NDArray[np.floating]
        array to be clipped and scaled
    max_scale : float, optional
        maximum scale factor, by default 255.0

    Returns
    -------
    npt.NDArray[np.floating]
        clipped and scaled array

    """
    # normalize to [0, 255]
    array = array - np.nanmin(array)
    max_val = np.nanmax(array)

    if max_val > 0:
        array = array / max_val

    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    return (array * max_scale).clip(0, max_scale)


def save_image_from_array(array: npt.NDArray[np.floating], path: str | Path) -> None:
    """Save a 2D array to a PNG/JPEG file.

    Parameters
    ----------
    array : np.ndarray
        2D array data to be saved to disk as png/jpeg format
    path : str | Path
        path to the output png/jpeg file to be saved

    """
    if Path(path).suffix.lower() not in _IMAGE_EXTENSIONS:
        msg = f"image extension must be one of {_IMAGE_EXTENSIONS}"
        raise ValueError(msg)
    array = np.asarray(array, dtype=np.float32)

    uint8_array = image_scale_clip(array).astype(np.uint8)

    Image.fromarray(uint8_array).save(path)
