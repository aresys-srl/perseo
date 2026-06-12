# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for perturbations/geodynamics/solid_tides.py core functionalities"""

import datetime

import numpy as np
from perseo_core.timing import PreciseDateTime

from perseo_perturbations.geodynamics.solid_tides import compute_displacement, compute_solid_earth_tides


class IERSSolidTidesTesting:
    """Testing iers_solid_tides core functionalities"""

    date = datetime.date(2021, 7, 5)
    lat_lon = (60, -32)
    _tolerance = 1e-9
    results = {
        "time": np.arange(0, 1441 * 60, 60),
        "north_first_10": np.array(
            [
                -0.015825623193191092,
                -0.015776041510506926,
                -0.015725788756351514,
                -0.015674868196914382,
                -0.01562328316260679,
                -0.015571037047455057,
                -0.015518133308917409,
                -0.015464575467705063,
                -0.015410367107603677,
                -0.015355511874798028,
                -0.015300013478574508,
            ]
        ),
        "north_last_10": np.array(
            [
                -0.01443196952059516,
                -0.014427569594715035,
                -0.014422307425932084,
                -0.014416182630802687,
                -0.014409194905818552,
                -0.014401344027576646,
                -0.014392629852808987,
                -0.014383052318342512,
                -0.014372611440315146,
                -0.01436130731786553,
            ]
        ),
        "east_first_10": np.array(
            [
                -0.006357325485034922,
                -0.006372752295016878,
                -0.00638771254767662,
                -0.006402199054716194,
                -0.006416204658214084,
                -0.0064297222316781494,
                -0.006442744680424984,
                -0.006455264942339754,
                -0.006467275988292951,
                -0.00647877082321224,
                -0.006489742484327297,
            ]
        ),
        "east_last_10": np.array(
            [
                -0.002393388238819648,
                -0.002421954240802441,
                -0.0024505107760917876,
                -0.0024790486791140924,
                -0.002507558779122499,
                -0.002536031901671232,
                -0.0025644588684377093,
                -0.0025928304989919077,
                -0.002621137609309824,
                -0.002649371019312525,
            ]
        ),
        "up_first_10": np.array(
            [
                -0.11601492659493721,
                -0.11608524317436182,
                -0.11615598508277784,
                -0.11622714587109889,
                -0.11629871897764008,
                -0.11637069773093842,
                -0.11644307534905124,
                -0.11651584494044266,
                -0.11658899950329762,
                -0.11666253192832363,
                -0.11673643499779585,
            ]
        ),
        "up_last_10": np.array(
            [
                -0.11936205377437703,
                -0.1193710318799089,
                -0.1193806438813301,
                -0.1193908901419048,
                -0.1194017708834717,
                -0.11941328618811609,
                -0.11942543599674635,
                -0.11943822010892563,
                -0.11945163818567663,
                -0.11946568974162648,
            ]
        ),
    }

    def test_compute_solid_earth_tides(self):
        """Testing solid earth tides core function"""
        times, displacement = compute_solid_earth_tides(
            year=self.date.year,
            month=self.date.month,
            day_of_month=self.date.day,
            lat_deg=self.lat_lon[0],
            lon_deg=self.lat_lon[1],
        )

        # comparing data to benchmark
        np.testing.assert_allclose(self.results["time"], times, atol=self._tolerance, rtol=0)

        # first 10 data of each displacement
        np.testing.assert_allclose(self.results["north_first_10"], displacement[:11, 0], atol=self._tolerance, rtol=0)
        np.testing.assert_allclose(self.results["east_first_10"], displacement[:11, 1], atol=self._tolerance, rtol=0)
        np.testing.assert_allclose(self.results["up_first_10"], displacement[:11, 2], atol=self._tolerance, rtol=0)

        # last 10 data of each displacement
        np.testing.assert_allclose(self.results["north_last_10"], displacement[-10:, 0], atol=self._tolerance, rtol=0)
        np.testing.assert_allclose(self.results["east_last_10"], displacement[-10:, 1], atol=self._tolerance, rtol=0)
        np.testing.assert_allclose(self.results["up_last_10"], displacement[-10:, 2], atol=self._tolerance, rtol=0)


class SolidTidesDisplacement:
    """Testing solid_tides.py core functionalities"""

    pt_pos = np.array(
        [
            (-2468789.77437779, -4626148.4320329, 3620025.27093258),
            (-2467963.52819618, -4626181.75345945, 3620542.41415808),
            (-2467068.51440511, -4626651.66776616, 3620552.45331852),
            (-2466645.20366593, -4626877.04176162, 3620552.92942237),
            (-2466140.52981391, -4627136.54745231, 3620565.21049583),
        ]
    )
    time = PreciseDateTime.from_utc_string("16-NOV-2019 04:06:56.329529000000")
    displacement_ref = np.array(
        [
            [0.06305905468150182, 0.05339999182683428, -0.049454763905432464],
            [0.0630511871586805, 0.0533805772901235, -0.04944733489296774],
            [0.06304064719973929, 0.05336377369371278, -0.04942786763966205],
            [0.0630356479034235, 0.0533558552189551, -0.04941857281234307],
            [0.06302973059997202, 0.05334632085770409, -0.04940773149090209],
        ]
    )

    def test_compute_displacement(self):
        """Testing compute_displacement function"""
        displacement = compute_displacement(target_xyz_coords=self.pt_pos, acquisition_time=self.time)

        np.testing.assert_array_almost_equal(displacement, self.displacement_ref, 1e-12)
