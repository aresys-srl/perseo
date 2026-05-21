# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""IGRF: legendre coefficient .shc file reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def read_shc(filename: str | Path) -> xr.Dataset:
    """Read IGRF spherical harmonic coefficient .shc file.

    Secular variation coefficients are already taken into account by the IGRF coefficients reported in the .shc file.

    Parameters
    ----------
    filename : str | Path
        filename of .shc file

    Returns
    -------
    xarray.Dataset
        Dataset with variables ``g`` and ``h``, dimensions ``time`` and ``coeff``. Coordinates include ``time``
        (datetime64), ``n`` (legendre function degree), and ``m`` (legendre function order).
    """

    lines = []
    with open(filename) as f:
        for line in f:
            if not line.startswith("#"):
                lines.append(line.split())

    # lines[0]: metadata — skip
    # lines[1]: year floats
    year_floats = [float(y) for y in lines[1]]
    time = np.array([f"{int(y)}-01-01" for y in year_floats], dtype="datetime64[D]")

    # lines[2:]: coefficient data
    coeff_dict = {}
    for tokens in lines[2:]:
        n, m = int(tokens[0]), int(tokens[1])
        coeff_dict[(n, m)] = np.array([float(v) for v in tokens[2:]])

    g_raw = {k: v for k, v in coeff_dict.items() if k[1] >= 0}
    h_raw = {(k[0], -k[1]): v for k, v in coeff_dict.items() if k[1] < 0}

    for key in [k for k in g_raw if k[1] == 0]:
        h_raw[key] = np.zeros_like(g_raw[key])

    if g_raw.keys() != h_raw.keys():
        raise RuntimeError("Malformed SHC file: g/h coefficient keys do not match")

    keys = sorted(g_raw.keys())
    g = np.column_stack([g_raw[k] for k in keys])
    h = np.column_stack([h_raw[k] for k in keys])

    n_vals = np.array([k[0] for k in keys])
    m_vals = np.array([k[1] for k in keys])

    return xr.Dataset(
        {"g": (["time", "coeff"], g), "h": (["time", "coeff"], h)},
        coords={
            "time": time,
            "n": ("coeff", n_vals),
            "m": ("coeff", m_vals),
        },
    )
