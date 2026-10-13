# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Save a KML/KMZ file with overlay data."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["save_image_overlay"]

KMZ_FILE_EXTENSION = ".kmz"
KML_FILE_EXTENSION = ".kml"
IMAGE_SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

KML_TEMPLATE = """
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
    <Document id="1">
        <GroundOverlay id="2">
            <name>Image Overlay</name>
            <color>__COLOR__</color>
            <Icon id="3">
                <href>__IMAGE_HREF__</href>
            </Icon>
            <gx:LatLonQuad>
                <coordinates>__COORDINATES__</coordinates>
            </gx:LatLonQuad>
        </GroundOverlay>
    </Document>
</kml>
"""


def save_image_overlay(
    image_path: str | Path,
    output_path: str | Path,
    corners: npt.NDArray[np.floating],
    opacity: float = 1.0,
) -> None:
    """Save an image as a KML/KMZ ground overlay providing image path and using LatLonQuad node.

    Parameters
    ----------
    image_path : str | Path
        path to the image file on disk
    output_path : str | Path
        output .kml or .kmz path
    corners : npt.NDArray[np.floating]
        image footprint corners, in the form (lon, lat) pairs, with shape (4, 2), with ccw ordering:
        bottom-left, bottom-right, top-right, top-left
    opacity : float
        image opacity, from 0.0 (transparent) to 1.0 (opaque).

    """
    image_path = Path(image_path).resolve()
    output_path = Path(output_path)
    is_kmz = str(output_path).endswith(KMZ_FILE_EXTENSION)

    if not is_kmz and not str(output_path).endswith(KML_FILE_EXTENSION):
        msg = "output_path must end with .kml or .kmz"
        raise ValueError(msg)

    if not 0.0 <= opacity <= 1.0:
        msg = "opacity must be between 0.0 and 1.0"
        raise ValueError(msg)

    _img_suffix = "." + str(image_path).split(".")[-1].lower()
    if _img_suffix not in IMAGE_SUPPORTED_EXTENSIONS:
        msg = f"image_path must point to a supported image file {IMAGE_SUPPORTED_EXTENSIONS}"
        raise ValueError(msg)

    # KML color format: aabbggrr (alpha, blue, green, red)
    alpha = round(opacity * 255)
    color_tag = f"{alpha:02x}ffffff"

    coords = [tuple(coord) for coord in corners]
    coords_str = " ".join(",".join(str(val) for val in point) for point in coords)

    kml_content = KML_TEMPLATE.replace("__COLOR__", color_tag).replace("__COORDINATES__", coords_str)

    if is_kmz:
        image_href = "files/" + image_path.name
    else:
        image_href = Path(os.path.relpath(str(image_path), output_path.parent)).as_posix()

    kml_content = kml_content.replace("__IMAGE_HREF__", image_href)

    if is_kmz:
        save_kmz(kml_content=kml_content, image_path=image_path, output_path=output_path)
    else:
        Path(output_path).write_text(kml_content.strip(), encoding="utf-8")


def save_kmz(kml_content: str, image_path: Path, output_path: Path) -> None:
    """Save KMZ archive with .kml file and overlay image.

    Parameters
    ----------
    kml_content : str
        kml content to be saved
    image_path : Path
        path to the image to be included in the archive
    output_path : Path
        kmz output path

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr("doc.kml", kml_content.strip())
        archive_image_path = f"files/{image_path.name}"
        kmz.write(image_path, arcname=archive_image_path)
