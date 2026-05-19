# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for the SHC coefficient file reader."""

from perseo_perturbations import igrf_14_coeff_path
from perseo_perturbations.geomagnetic.core.shc_reader import read_shc


class TestShcReader:
    """Tests for the SHC reader."""

    def test_returns_dataset(self):
        """Reader returns an xarray Dataset with expected variables and coords."""
        ds = read_shc(str(igrf_14_coeff_path))
        assert "g" in ds
        assert "h" in ds
        assert "n" in ds.coords
        assert "m" in ds.coords
        assert "time" in ds.coords

    def test_coeff_shapes(self):
        """g and h have identical (time, coeff) dimensions and sizes."""
        ds = read_shc(str(igrf_14_coeff_path))
        assert ds["g"].dims == ("time", "coeff")
        assert ds["h"].dims == ("time", "coeff")
        assert ds["g"].shape == ds["h"].shape

    def test_degrees_and_orders(self):
        """Coefficient degrees are 1-13 and order never exceeds degree."""
        ds = read_shc(str(igrf_14_coeff_path))
        n_vals = ds["n"].values
        m_vals = ds["m"].values
        assert all(1 <= n <= 13 for n in n_vals)
        assert all(0 <= m <= n for n, m in zip(n_vals, m_vals, strict=True))
