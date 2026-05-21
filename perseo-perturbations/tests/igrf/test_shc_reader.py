# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for the SHC coefficient file reader."""

import unittest

from perseo_perturbations import igrf_14_coeff_path
from perseo_perturbations.geomagnetic.core.shc_reader import read_shc


class TestShcReader(unittest.TestCase):
    """Tests for the SHC reader."""

    def test_returns_dataset(self):
        """Reader returns an xarray Dataset with expected variables and coords."""
        ds = read_shc(str(igrf_14_coeff_path))
        self.assertIn("g", ds)
        self.assertIn("h", ds)
        self.assertIn("n", ds.coords)
        self.assertIn("m", ds.coords)
        self.assertIn("time", ds.coords)

    def test_coeff_shapes(self):
        """g and h have identical (time, coeff) dimensions and sizes."""
        ds = read_shc(str(igrf_14_coeff_path))
        self.assertEqual(ds["g"].dims, ("time", "coeff"))
        self.assertEqual(ds["h"].dims, ("time", "coeff"))
        self.assertEqual(ds["g"].shape, ds["h"].shape)

    def test_degrees_and_orders(self):
        """Coefficient degrees are 1-13 and order never exceeds degree."""
        ds = read_shc(str(igrf_14_coeff_path))
        n_vals = ds["n"].values
        m_vals = ds["m"].values
        self.assertTrue(all(1 <= n <= 13 for n in n_vals))
        self.assertTrue(all(0 <= m <= n for n, m in zip(n_vals, m_vals, strict=True)))


if __name__ == "__main__":
    unittest.main()
